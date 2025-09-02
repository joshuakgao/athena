"""Encodes a fen and uci move into a AlphaZero-style input tensor."""

import chess
import numpy as np

from athena.encoders._base_encoder import BaseEncoder


class ActionEncoder(BaseEncoder):
    """Encodes a fen and uci move into a AlphaZero-style input tensor."""

    def __init__(self, cfg):
        """Initialize the ActionEncoder with the given configuration.

        Args:
            cfg (Config): Configuration object containing encoder settings.
        """
        assert cfg.encoder.input_encoder.type == "action", (
            "ActionEncoder should only be used with action input encoder type."
        )

        super().__init__()
        self.input_channels = cfg.encoder.input_encoder.input_channels

    def encode(self, fen, move_uci):
        """Convert a FEN and move into an AlphaZero-style input tensor with move encoding.

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
        board_tensor = np.zeros((8, 8, self.input_channels), dtype=np.float32)

        # Split the FEN into its components
        parts = fen.split()
        board_part = parts[0]
        color_part = parts[1]
        castling_part = parts[2]
        en_passant_part = parts[3]
        halfmove_part = int(parts[4])
        # fullmove_part = int(parts[5])

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

    def decode(self, board_tensor):
        """Decode the board tensor into FEN and UCI move."""
        raise NotImplementedError("ActionEncoder does not support decoding.")
