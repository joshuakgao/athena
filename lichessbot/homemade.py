import logging
import os
import random
import re
import sys

import chess
import numpy as np
import torch
from chess.engine import Limit, PlayResult
from lichessbot.lib.engine_wrapper import MinimalEngine
from lichessbot.lib.lichess_types import HOMEMADE_ARGS_TYPE, MOVE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from architecture import Athena
from utils.chess_utils import column_letter_to_num, is_fen_valid, is_uci_valid

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


def encode_input(fens):
    """Encode a batch of FEN strings into a batch of input tensors with shape (batch_size, 9, 8, 8)."""
    batch_size = len(fens)
    inputs = np.zeros((batch_size, 9, 8, 8), dtype=np.float32)

    types = ["p", "n", "b", "r", "q", "k"]
    for i, fen in enumerate(fens):
        assert is_fen_valid(fen)
        board = chess.Board(fen)

        # Encode board position for each piece type
        for t_idx, type in enumerate(types):
            s = str(board)
            s = re.sub(f"[^{type}{type.upper()} \n]", ".", s)
            s = re.sub(f"{type}", "-1", s)
            s = re.sub(f"{type.upper()}", "1", s)
            s = re.sub(f"\.", "0", s)
            board_mat = [[int(x) for x in row.split(" ")] for row in s.split("\n")]
            inputs[i, t_idx, :, :] = np.array(board_mat)

        fen_split = fen.split(" ")
        turn, castling_rights, enpassant_rights = (
            fen_split[1],
            fen_split[2],
            fen_split[3],
        )

        # Encode turn
        inputs[i, 6, :, :] = 1 if turn == "w" else -1

        # Encode castling rights
        castling_tensor = np.zeros((8, 8), dtype=np.int8)

        # Define the squares for queenside and kingside castling for both white and black
        white_queenside = (7, 0)
        white_kingside = (7, 7)
        black_queenside = (0, 0)
        black_kingside = (0, 7)

        if turn == "w":
            # If white can castle kingside, set b1 to 1
            if "K" in castling_rights:
                castling_tensor[white_kingside] = 1
            # If white can castle queenside, set g1 to 1
            if "Q" in castling_rights:
                castling_tensor[white_queenside] = 1

        if turn == "b":
            # If black can castle kingside, set b8 to -1
            if "k" in castling_rights:
                castling_tensor[black_kingside] = -1

            # If black can castle queenside, set g8 to -1
            if "q" in castling_rights:
                castling_tensor[black_queenside] = -1

        # Store the castling tensor at the correct place in the inputs array
        inputs[i, 7, :, :] = castling_tensor

        # Encode en passant rights
        enpassant_tensor = np.zeros((8, 8), dtype=np.int8)
        if enpassant_rights != "-":
            file = ord(enpassant_rights[0]) - ord("a")
            rank = 8 - int(enpassant_rights[1])
            enpassant_tensor[rank, file] = 1 if rank == 5 else -1
        inputs[i, 8, :, :] = enpassant_tensor

    return torch.tensor(inputs)


def encode_output(ucis):
    """Encode a batch of UCI moves to a batch of output tensors with shape (batch_size, 2, 8, 8)."""
    batch_size = len(ucis)
    outputs = np.zeros((batch_size, 2, 8, 8), dtype=np.float32)

    for i, uci in enumerate(ucis):
        assert is_uci_valid(uci)

        # "From" square
        from_row, from_col = 8 - int(uci[1]), column_letter_to_num(uci[0])
        outputs[i, 0, from_row, from_col] = 1

        # "To" square
        to_row, to_col = 8 - int(uci[3]), column_letter_to_num(uci[2])
        outputs[i, 1, to_row, to_col] = 1

    return torch.tensor(outputs)


def decode_input(tensor):
    """Decode a batch of input tensors (shape [batch_size, 9, 8, 8]) back to FEN strings."""
    batch_size = tensor.shape[0]
    fens = []

    # Piece types in FEN order (pawn, knight, bishop, rook, queen, king)
    piece_types = ["p", "n", "b", "r", "q", "k"]

    for i in range(batch_size):
        board = chess.Board(None)  # Create empty board
        board.clear()  # Remove all pieces

        # Decode pieces (channels 0-5)
        for t_idx, piece in enumerate(piece_types):
            layer = tensor[i, t_idx].numpy()
            for rank in range(8):
                for file in range(8):
                    val = layer[rank, file]
                    if val > 0.5:  # White piece
                        board.set_piece_at(
                            chess.square(file, 7 - rank),
                            chess.Piece.from_symbol(piece.upper()),
                        )
                    elif val < -0.5:  # Black piece
                        board.set_piece_at(
                            chess.square(file, 7 - rank),
                            chess.Piece.from_symbol(piece),
                        )

        # Decode turn (channel 6)
        turn = "w" if tensor[i, 6, 0, 0] > 0 else "b"

        # Decode castling rights (channel 7)
        castling = ""
        castling_layer = tensor[i, 7].numpy()

        # Check white castling
        if castling_layer[7, 7] > 0.5:  # White kingside (h1)
            castling += "K"
        if castling_layer[7, 0] > 0.5:  # White queenside (a1)
            castling += "Q"

        # Check black castling
        if castling_layer[0, 7] < -0.5:  # Black kingside (h8)
            castling += "k"
        if castling_layer[0, 0] < -0.5:  # Black queenside (a8)
            castling += "q"

        castling = castling if castling else "-"

        # Decode en passant (channel 8)
        enpassant_layer = tensor[i, 8].numpy()
        ep_square = "-"
        for rank in range(8):
            for file in range(8):
                if abs(enpassant_layer[rank, file]) > 0.5:
                    ep_square = chr(ord("a") + file) + str(8 - rank)
                    break

        # Construct FEN
        fen = f"{board.board_fen()} {turn} {castling} {ep_square} 0 1"
        fens.append(fen)

    return fens


def decode_output(tensor):
    """Decode policy output tensor (shape [2, 8, 8]) to a single UCI move."""
    # Get from and to squares
    from_layer = tensor[0].detach().cpu().numpy()
    to_layer = tensor[1].detach().cpu().numpy()

    from_square = np.unravel_index(np.argmax(from_layer), (8, 8))
    to_square = np.unravel_index(np.argmax(to_layer), (8, 8))

    # Convert to UCI notation
    from_file = chr(ord("a") + from_square[1])
    from_rank = str(8 - from_square[0])
    to_file = chr(ord("a") + to_square[1])
    to_rank = str(8 - to_square[0])

    return f"{from_file}{from_rank}{to_file}{to_rank}"


class AthenaEngine(MinimalEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = "checkpoints/best_model_1.11_resnet19.pt"
        self.model = Athena(device="cpu")
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.temperature = 0.8
        self.min_temp = 0.3

    def search(self, board: chess.Board, time_limit: Limit = None, *args):
        input_tensor = encode_input([board.fen()]).to(self.model.device)
        with torch.no_grad():
            policy = self.model(input_tensor)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return PlayResult(None, None)

        best_move = self._select_best_move(legal_moves, policy)
        return PlayResult(best_move, None)

    def _select_best_move(self, legal_moves, policy):
        # Flatten the policy layers
        policy_from_flat = policy[0, 0].flatten()
        policy_to_flat = policy[0, 1].flatten()

        # Apply softmax to the flattened policy layers
        policy_from = torch.softmax(policy_from_flat, dim=-1).reshape(8, 8)
        policy_to = torch.softmax(policy_to_flat, dim=-1).reshape(8, 8)

        # Score each legal move
        move_scores = []
        for move in legal_moves:
            from_sq = (7 - (move.from_square // 8), move.from_square % 8)
            to_sq = (7 - (move.to_square // 8), move.to_square % 8)
            score = policy_from[from_sq[0], from_sq[1]] * policy_to[to_sq[0], to_sq[1]]
            move_scores.append(score.item())

        # Select move with highest score
        best_move = legal_moves[np.argmax(move_scores)]
        logger.info(best_move)
        return best_move


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""
