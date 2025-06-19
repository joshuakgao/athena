import chess
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.device_selector import device_selector


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Skip connection
        return F.relu(x)


class AthenaResnet(nn.Module):
    def __init__(
        self,
        input_channels=19,
        width=256,
        num_blocks=19,
        K=128,
        M=32,
        device="auto",
    ):
        super(AthenaResnet, self).__init__()
        self.device = device_selector(device, label="Athena")
        self.K = K
        self.M = M
        self.output_bins = (
            K + 2 * M + 1
        )  # K for win probs, 2*M for mate-for and mate-against, 1 for checkmate

        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)

        # Residual stack
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(width) for _ in range(num_blocks)]
        )

        # Value head
        self.value_conv1 = nn.Conv2d(width, 32, kernel_size=1)
        self.value_bn1 = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 512)
        self.value_fc2 = nn.Linear(512, self.output_bins)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.residual_blocks(x)

        value_x = F.relu(self.value_bn1(self.value_conv1(x)))
        value_x = value_x.view(value_x.size(0), -1)
        value_x = F.relu(self.value_fc1(value_x))
        value_logits = self.value_fc2(value_x)

        return value_logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_action_value(self, fen, move_uci, input_channels=24):
        """
        Convert a FEN and move into an AlphaZero-style input tensor with move encoding.

        Args:
            fen (str): The FEN string representing the chess position.
            move_uci (str): The UCI string representing the move to be made.

        Returns:
            np.ndarray: A 8x8xN tensor where:
                    - Planes 0-17: Board state encoding (as before)
                    - Plane 18: 'From' square of the move (1 where piece moves from)
                    - Plane 19: 'To' square of the move (1 where piece moves to)
                    - Plane 20: Promote to Queen (entire plane 1 if promotion)
                    - Plane 21: Promote to Rook
                    - Plane 22: Promote to Bishop
                    - Plane 23: Promote to Knight
        """
        board = chess.Board(fen)
        color_to_move = board.turn

        # Initialize tensor with extra planes for move encoding
        board_tensor = np.zeros((8, 8, input_channels), dtype=np.float32)

        # Split the FEN into its components
        parts = fen.split()
        board_part = parts[0]
        color_part = parts[1]
        castling_part = parts[2]
        en_passant_part = parts[3]
        halfmove_part = int(parts[4])
        fullmove_part = int(parts[5])

        # Piece encoding (planes 0-11)
        piece_to_plane = {
            "P": 0,
            "N": 1,
            "B": 2,
            "R": 3,
            "Q": 4,
            "K": 5,
            "p": 6,
            "n": 7,
            "b": 8,
            "r": 9,
            "q": 10,
            "k": 11,
        }

        # Parse the board
        row = 0
        col = 0
        for c in board_part:
            if c == "/":
                row += 1
                col = 0
            elif c.isdigit():
                col += int(c)
            else:
                plane = piece_to_plane[c]
                board_tensor[row, col, plane] = 1
                col += 1

        # Set castling rights (planes 12-15: K, Q, k, q)
        castling_map = {
            "w": {"K": 12, "Q": 13, "k": 14, "q": 15},
            "b": {"k": 12, "q": 13, "K": 14, "Q": 15},
        }
        for right, plane in castling_map[color_part].items():
            if right in castling_part:
                board_tensor[:, :, plane] = 1

        # 50-move rule (plane 16)
        board_tensor[:, :, 16] = min(halfmove_part / 50.0, 1.0)

        # En passant (plane 17)
        if en_passant_part != "-":
            ep_col = ord(en_passant_part[0]) - ord("a")
            ep_row = 8 - int(en_passant_part[1])
            board_tensor[ep_row, ep_col, 17] = 1

        # Move encoding (planes 18-23)
        move = chess.Move.from_uci(move_uci)

        # Compute coordinates without perspective flip
        from_row = 7 - (move.from_square // 8)
        from_col = move.from_square % 8
        to_row = 7 - (move.to_square // 8)
        to_col = move.to_square % 8

        board_tensor[from_row, from_col, 18] = 1  # From square
        board_tensor[to_row, to_col, 19] = 1  # To square

        # Promotion planes (20-23)
        if move.promotion:
            if move.promotion == chess.QUEEN:
                board_tensor[:, :, 20] = 1
            elif move.promotion == chess.ROOK:
                board_tensor[:, :, 21] = 1
            elif move.promotion == chess.BISHOP:
                board_tensor[:, :, 22] = 1
            elif move.promotion == chess.KNIGHT:
                board_tensor[:, :, 23] = 1

        # Flip the board if black to move (only rows)
        if color_to_move == chess.BLACK:
            board_tensor = np.flip(board_tensor, axis=0).copy()

        return board_tensor

    def encode_win_prob(self, win_prob, mate):
        """
        Encode win probability and mate information into a tensor with K + 2*M + 1 bins.
        We need the extra bin for the move that checkmates the opponent.
        This extra bin isn't shown on the negative side of the tensor, since a move that checkmates yourself is illegal.
        Bin structure:
        [-M1, ..., -M20, win_prob_bins..., M20, ..., M1, M0 (Move that checkmates opponent)]

        Args:
            win_prob (float): Win probability in [0, 1].
            mate (int): Number of plies to mate. Positive = mate for, negative = mate against.
            K (int): Number of win probability bins.
            M (int): Number of mate bins on each side.

        Returns:
            np.ndarray: One-hot encoded tensor of shape (K + 2*M + 1,)
        """
        # assert 0.0 <= win_prob <= 1.0

        tensor = np.zeros((self.K + 2 * self.M + 1,), dtype=np.float32)

        if isinstance(mate, str):
            # assert mate in ("#", "-"), f"Unrecognized mate string: {mate}"
            if mate == "#":
                assert win_prob == 1.0
                index = -1
            elif mate == "-":
                index = self.M + int(round(win_prob * (self.K - 1)))
        else:
            # assert mate != 0
            if mate > 0:
                assert win_prob == 1.0
                index = self.K + 2 * self.M - min(mate, self.M)
            elif mate < 0:
                assert win_prob == 0.0
                index = self.M - min(-mate, self.M)
            else:
                raise ValueError(f"Invalid mate value: {mate}")

        # assert -1 <= index < len(tensor), f"Index {index} out of bounds"
        tensor[index] = 1.0
        return tensor

    def decode_win_prob_bins(self, tensor):
        assert len(tensor) == self.output_bins
        index = np.argmax(tensor)

        if index == self.output_bins - 1:
            return 1.0, "#"
        elif index < self.M:
            return 0.0, -(self.M - index)
        elif index < self.M + self.K:
            win_prob = (index - self.M) / (self.K - 1)
            return win_prob, "-"
        else:
            # Positive mate bins (inverted order)
            mate = self.K + 2 * self.M - index
            return 1.0, mate
