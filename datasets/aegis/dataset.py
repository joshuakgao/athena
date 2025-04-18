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

    def decode_move_policy(self, fens, policies):
        return _decode_move_policy(fens, policies)


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
        ]
        self.metadata_path = self.dir / "metadata.json"
        self.data = pd.DataFrame(columns=["fen", "history", "best_move"])
        self.n = n

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.sample_dataset()

    def sample_dataset(self):
        # Clear existing data
        self.data = pd.DataFrame(columns=["fen", "history", "best_move"])
        train_set_size = self.metadata["train_set_size"]

        # Don't divide by 0
        assert train_set_size > 0

        # Get percentage of dataset we want, based on self.n
        sampling_rate = self.n / train_set_size

        rows = []
        for shard_path in tqdm(self.shard_paths, desc=f"Sampling ~{self.n} of Aegis"):
            df = pd.read_parquet(shard_path)
            for _, row in df.iterrows():
                if random.random() < sampling_rate:
                    rows.append(
                        {
                            "fen": row["fen"],
                            "history": row["history"],
                            "best_move": row["best_move"],
                            "eval": row["eval"],
                        }
                    )
        self.data = pd.DataFrame(rows)

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
        v = _encode_centipawn([centipawn])
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
        self.data = pd.DataFrame(columns=["fen", "history", "best_move"])

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
        x = _encode_position([fen], [history])
        y = _encode_move([best_move])
        v = _encode_centipawn([centipawn])
        return x[0], y[0], v[0], fen, best_move, centipawn


def _encode_position(fens, histories):
    """
    Encode a batch of FEN strings into a batch of input tensors with shape (batch_size, 59, 8, 8).
    6 layers for each piece type (pawn, knight, bishop, rook, queen, king), 1 layer for castling rights
    8 fens are then encoded in this manner (1 current + 7 history),
    1 layer for repetition count
    1 layer is a constant 1s tensor to indicate the board area to reduce the effect of edge blurring in the convlutional layers.
    1 layer is for who's turn it is
    """
    batch_size = len(fens)
    inputs = np.zeros((batch_size, 59, 8, 8), dtype=np.float32)

    types = ["p", "n", "b", "r", "q", "k"]
    for i, (fen, history) in enumerate(zip(fens, histories)):
        history = list(history)
        while len(history) < 7:
            history.append(None)

        board = chess.Board(fen)
        layer_idx = 0

        history.insert(0, fen)
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

        # Encode repetition
        repetition_count = 0
        if board.is_repetition(count=1):
            repetition_count = 1
        elif board.is_repetition(count=2):
            repetition_count = 2
        elif board.is_repetition(count=3):
            repetition_count = 3
        elif board.is_repetition(count=4):
            repetition_count = 4
        elif board.is_repetition(count=5):
            repetition_count = 5
        elif board.is_repetition(count=6):
            repetition_count = 6
        elif board.is_repetition(count=7):
            repetition_count = 7
        inputs[i, layer_idx, :, :] = np.full((8, 8), repetition_count, dtype=np.float32)
        layer_idx += 1

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


def _encode_centipawn(evals):
    centipawn_evals = np.tanh(np.array(evals, dtype=np.float32))
    return torch.tensor(centipawn_evals).unsqueeze(1)


def _encode_move(ucis):
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


def _decode_move_policy(policies, fens):
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
