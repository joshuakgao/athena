import json
import os
import random
import re
import sys
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.chess_utils import column_letter_to_num, is_uci_valid


class AegisDataset:
    def __init__(self, dir="datasets/aegis/data", train_n=10_000_000):
        self.train_dataset = AegisTrainDataset(dir, n=train_n)
        self.test_dataset = AegisTestDataset(dir)

    def encode_position(self, fens, histories):
        return _encode_position(fens, histories)

    def encode_move(self, ucis):
        return _encode_move(ucis)

    def encode_eval(self, evals):
        return _encode_eval(evals)

    def decode_move(self, fens, policies):
        return _decode_move(fens, policies)

    def decode_eval(self, encoded_evals):
        return _decode_eval(encoded_evals)


class AegisTrainDataset(Dataset):
    def __init__(self, dir="datasets/aegis/data", n=10_000_000):
        """
        Args:
            dir: Directory containing the dataset
        """
        self.dir = Path(dir)
        # Get all shards except test.parquet
        self.shard_paths = [
            p for p in self.dir.glob("*.parquet") if p.name != "test.parquet"
        ][:1]
        self.metadata_path = self.dir / "metadata.json"
        self.data = pd.DataFrame(columns=["fen", "history", "best_move", "eval"])
        self.n = n

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.sample_dataset()

    def sample_dataset(self):
        # Clear existing data
        self.data = pd.DataFrame(columns=["fen", "history", "best_move", "eval"])
        num_shards = len(self.shard_paths)
        samples_per_shard = self.n // num_shards

        pieces = []
        for shard_path in tqdm(self.shard_paths, desc=f"Sampling ~{self.n} of Aegis"):
            df = pd.read_parquet(
                shard_path, columns=["fen", "history", "best_move", "eval"]
            )
            num_rows = df.shape[0]
            random_indexes = random.sample(
                range(num_rows), min(samples_per_shard, num_rows)
            )
            sampled_df = df.iloc[random_indexes]
            pieces.append(sampled_df)

        self.data = pd.concat(pieces, ignore_index=True)

    def __len__(self):
        # len of dataset, not of entire dataset
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row["fen"]
        history = list(row["history"])
        best_move = row["best_move"]
        centipawn = row["eval"]
        x = _encode_position([fen], [history])
        y = _encode_move([best_move])
        v = _encode_eval([centipawn])
        return x[0], y[0], v[0]


class AegisTestDataset(Dataset):
    def __init__(self, dir="datasets/aegis/data"):
        """
        Args:
            dir: Directory containing the dataset
        """
        self.dir = Path(dir)
        self.test_path = self.dir / "test.parquet"
        self.metadata_path = self.dir / "metadata.json"
        self.data = pd.DataFrame(
            columns=["fen", "history", "best_move", "elo", "bot", "depth"]
        )

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Load test set
        df = pd.read_parquet(self.test_path)
        rows = []
        for _, row in tqdm(df.iterrows(), desc="Loading Aegis test set"):
            rows.append(
                {
                    "fen": row["fen"],
                    "history": row["history"],
                    "best_move": row["best_move"],
                    "eval": row["eval"],
                    "bot": row["bot"],
                    "depth": row["depth"],
                    "elo": row["elo"],
                }
            )
        self.data = pd.DataFrame(rows)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row["fen"]
        history = list(row["history"])
        best_move = row["best_move"]
        centipawn = row["eval"]
        bot = row["bot"]
        depth = row["depth"]
        elo = row["elo"]
        eval = row["eval"]

        x = _encode_position([fen], [history])
        y = _encode_move([best_move])
        v = _encode_eval([centipawn])
        return x[0], y[0], v[0], fen, bot, depth, elo, eval


def _encode_position(fens, histories):
    """
    Encode a batch of FEN strings into a batch of input tensors with shape (batch_size, 10, 8, 8).
    6 layers for each piece type (pawn, knight, bishop, rook, queen, king), 1 layer for castling rights
    8 fens are then encoded in this manner (1 current + 7 history),
    1 layer for repetition count
    1 layer is a constant 1s tensor to indicate the board area to reduce the effect of edge blurring in the convlutional layers.
    1 layer is for who's turn it is
    """
    batch_size = len(fens)
    inputs = np.zeros((batch_size, 10, 8, 8), dtype=np.float32)

    types = ["p", "n", "b", "r", "q", "k"]
    for i, (fen, history) in enumerate(zip(fens, histories)):
        history = list(history)
        while len(history) < 7:
            history.append(None)

        board = chess.Board(fen)
        layer_idx = 0

        # history.insert(0, fen)
        history = [fen]
        for _fen in history:
            _board = chess.Board(_fen)
            if _fen == None:
                for _ in range(7):
                    inputs[i, layer_idx, :, :] = np.zeros((8, 8), dtype=np.float32)
                    layer_idx += 1
                continue
            # Encode board position for each piece type
            for type in types:
                s = str(_board)
                s = re.sub(f"[^{type}{type.upper()} \n]", ".", s)
                s = re.sub(f"{type}", "-1", s)
                s = re.sub(f"{type.upper()}", "1", s)
                s = re.sub(f"\.", "0", s)
                board_mat = [[int(x) for x in row.split(" ")] for row in s.split("\n")]
                inputs[i, layer_idx, :, :] = np.array(board_mat)
                layer_idx += 1

            # Encode castling rights
            castling_tensor = np.zeros((8, 8), dtype=np.float32)
            if board.has_kingside_castling_rights(chess.WHITE):
                castling_tensor[7, 7] = 1
            if board.has_queenside_castling_rights(chess.WHITE):
                castling_tensor[7, 0] = 1
            if board.has_kingside_castling_rights(chess.BLACK):
                castling_tensor[0, 7] = -1
            if board.has_queenside_castling_rights(chess.BLACK):
                castling_tensor[0, 0] = -1
            inputs[i, layer_idx, :, :] = castling_tensor
            layer_idx += 1

            # Encode En Passant
            en_passant_tensor = np.zeros((8, 8), dtype=np.float32)
            if board.ep_square is not None:
                row, col = divmod(board.ep_square, 8)
                en_passant_tensor[row, col] = 1
            inputs[i, layer_idx, :, :] = en_passant_tensor
            layer_idx += 1

        # # Encode repetition
        # repetition_count = 0
        # if board.is_repetition(count=1):
        #     repetition_count = 1
        # elif board.is_repetition(count=2):
        #     repetition_count = 2
        # elif board.is_repetition(count=3):
        #     repetition_count = 3
        # elif board.is_repetition(count=4):
        #     repetition_count = 4
        # elif board.is_repetition(count=5):
        #     repetition_count = 5
        # elif board.is_repetition(count=6):
        #     repetition_count = 6
        # elif board.is_repetition(count=7):
        #     repetition_count = 7
        # inputs[i, layer_idx, :, :] = np.full((8, 8), repetition_count, dtype=np.float32)
        # layer_idx += 1

        # Encode turn
        turn = board.turn
        if turn == chess.WHITE:
            inputs[i, layer_idx, :, :] = 1
        else:
            inputs[i, layer_idx, :, :] = -1
        layer_idx += 1

        # Encode board
        inputs[i, layer_idx, :, :] = np.ones((8, 8), dtype=np.float32)
        layer_idx += 1

    return torch.tensor(inputs)


def _encode_eval(evals, scale=400):
    """
    Scales and applies tanh to centipawn evaluations.
    Maps typical range [-1000, 1000] into ~[-0.999, 0.999]
    """
    scaled = np.array(evals, dtype=np.float64) / scale
    squashed = np.tanh(scaled)
    return torch.tensor(squashed, dtype=torch.float32).unsqueeze(1)


def _encode_move(ucis):
    """Encode moves into 73×8×8 format."""
    batch_size = len(ucis)
    outputs = torch.zeros((batch_size, 73, 8, 8))

    for i, uci in enumerate(ucis):
        move = chess.Move.from_uci(uci)
        plane_idx = _get_move_type_index(move)
        from_row = 7 - (move.from_square // 8)
        from_col = move.from_square % 8
        outputs[i, plane_idx, from_row, from_col] = 1

    return outputs


def _get_move_type_index(move: chess.Move) -> int:
    """Convert a chess move to its 73-channel plane index."""
    # ─── Queen-like Moves (0-55) ──────────────────────────────────
    dx = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dy = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)

    # 8 directions × 7 distances
    if abs(dx) == abs(dy):  # Diagonal
        direction = {
            (1, 1): 0,  # NE
            (-1, 1): 1,  # NW
            (1, -1): 2,  # SE
            (-1, -1): 3,  # SW
        }[(dx // abs(dx), dy // abs(dy))]
        distance = abs(dx) - 1  # 0-6
        return direction * 7 + distance

    elif dx == 0 or dy == 0:  # Straight
        direction = {
            (0, 1): 4,  # N
            (0, -1): 5,  # S
            (1, 0): 6,  # E
            (-1, 0): 7,  # W
        }[(dx // max(1, abs(dx)), dy // max(1, abs(dy)))]
        distance = (abs(dx) + abs(dy)) - 1  # 0-6
        return direction * 7 + distance

    # ─── Knight Moves (56-63) ──────────────────────────────────────
    elif {abs(dx), abs(dy)} == {1, 2}:
        knight_directions = [
            (1, 2),
            (2, 1),
            (-1, 2),
            (-2, 1),
            (1, -2),
            (2, -1),
            (-1, -2),
            (-2, -1),
        ]
        return 56 + knight_directions.index((dx, dy))

    # ─── Underpromotions (64-72) ───────────────────────────────────
    elif move.promotion and move.promotion != chess.QUEEN:
        promotion_type = {
            chess.KNIGHT: 0,
            chess.BISHOP: 1,
            chess.ROOK: 2,
        }[move.promotion]

        # Check if capture (left/right) or push (forward)
        if dx == -1:  # Left capture (e.g., a7→b8)
            direction = 0
        elif dx == 1:  # Right capture (e.g., h7→g8)
            direction = 1
        else:  # Forward push (e.g., e7→e8)
            direction = 2

        return 64 + promotion_type * 3 + direction

    raise ValueError(f"Unclassified move: {move}")


def _decode_move(policies, fens):
    """Convert 73×8×8 policy to UCI moves."""
    best_moves = []
    for policy, fen in zip(policies, fens):
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Score each legal move using policy
        move_scores = []
        for move in legal_moves:
            plane_idx = _get_move_type_index(move)
            from_row, from_col = 7 - (move.from_square // 8), move.from_square % 8
            score = policy[plane_idx, from_row, from_col]
            move_scores.append(score.item())

        best_move = legal_moves[np.argmax(move_scores)]
        best_moves.append(best_move)
    return best_moves


def _decode_eval(encoded_evals, scale=400):
    """
    Inverts tanh and rescales the values to recover centipawn evaluations.
    """
    if not isinstance(encoded_evals, torch.Tensor):
        encoded_evals = torch.tensor(encoded_evals)
    encoded_evals = encoded_evals.detach().cpu().numpy()
    recovered = np.arctanh(encoded_evals)
    return (recovered * scale).tolist()
