import os
import re
import sys

import chess
import numpy as np
import torch
from chess.engine import Limit, PlayResult

from lichessbot.lib.engine_wrapper import MinimalEngine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from architecture import Athena
from alphazero_arch import AlphaZeroNet


def encode_input(fens):
    """Encode a batch of FEN strings into a batch of input tensors with shape (batch_size, 9, 8, 8)."""
    batch_size = len(fens)
    inputs = np.zeros((batch_size, 9, 8, 8), dtype=np.float32)

    types = ["p", "n", "b", "r", "q", "k"]
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        assert board.is_valid()

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


def decode_move(policies, fens):
    """Decode policy output tensor (shape [2, 8, 8]) to a single UCI move."""
    best_moves = []
    for policy, fen in zip(policies, fens):
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Check policy tensor shape
        assert policy.shape == (2, 8, 8)

        # Flatten the policy layers
        policy_from_flat = policy[0].flatten()
        policy_to_flat = policy[1].flatten()

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
        best_moves.append(best_move)
    return best_moves


class AthenaEngine(MinimalEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = "checkpoints/best_model_1.11_resnet19.pt"
        # self.model = AlphaZeroNet(device="cpu")
        self.model = Athena(device="cpu")
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.temperature = 0.1

    def search(self, board: chess.Board, time_limit: Limit = None, *args):
        input_tensor = encode_input([board.fen()]).to(self.model.device)
        with torch.no_grad():
            policy = self.model(input_tensor)

        best_move = decode_move(policy[0], board.fen())
        return PlayResult(best_move, None)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""
