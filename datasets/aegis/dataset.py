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

    def decode_move(self, policies, fens):
        return _decode_move(policies, fens)

    def decode_eval(self, encoded_evals):
        return _decode_eval(encoded_evals)

    def flat_index_of_move(self, move):
        return _flat_index_of_move(move)


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
        ][:3]
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
    AlphaZero-style encoding → Tensor  (B, 119, 8, 8)

    8 half-moves (current + up to 7 previous):
        • 6 planes  : white   {P,N,B,R,Q,K}
        • 6 planes  : black   {p,n,b,r,q,k}
        • 2 planes  : repetition flags
                      plane-0 = 1  if position seen once before
                      plane-1 = 1  if position seen ≥2× before
    Static context (last 7 planes):
        112 : side-to-move           (all-1 if White else all-0)
        113 : white K-side castling  (all-1 / all-0)
        114 : white Q-side castling
        115 : black K-side castling
        116 : black Q-side castling
        117 : full-move number / 100   (float)
        118 : half-move clock  / 100   (float)
    """
    B = len(fens)
    planes = np.zeros((B, 119, 8, 8), dtype=np.float32)

    piece_symbols = "PNBRQKpnbrqk"  # order is important

    for b_idx, (fen, hist) in enumerate(zip(fens, histories)):
        # -------- gather up to 8 FENs (current + 7 history) ----------
        fen_stack = [fen] + list(hist)[:7]
        fen_stack += [None] * (8 - len(fen_stack))  # pad with Nones

        for t, fen_t in enumerate(fen_stack):
            base_plane = t * 14  # 0,14,…,98
            if fen_t is None:
                continue  # leave zeros

            board_t = chess.Board(fen_t)

            # --- 12 piece-presence planes  --------------------------------
            s = str(board_t).replace("\n", " ")
            squares = np.array(s.split(), dtype="<U1").reshape(8, 8)

            for p_idx, symbol in enumerate(piece_symbols):
                mask = (squares == symbol).astype(np.float32)
                planes[b_idx, base_plane + p_idx, :, :] = mask

            # --- repetition planes (only reliable for *current* board) ----
            if t == 0:
                if board_t.is_repetition(1):
                    planes[b_idx, base_plane + 12, :, :] = 1
                if board_t.is_repetition(2):
                    planes[b_idx, base_plane + 13, :, :] = 1

        # ==================================================================
        # static context (planes 112…118)
        board0 = chess.Board(fen)  # current board

        planes[b_idx, 112, :, :] = 1.0 if board0.turn == chess.WHITE else 0.0

        planes[b_idx, 113, :, :] = (
            1.0 if board0.has_kingside_castling_rights(chess.WHITE) else 0.0
        )
        planes[b_idx, 114, :, :] = (
            1.0 if board0.has_queenside_castling_rights(chess.WHITE) else 0.0
        )
        planes[b_idx, 115, :, :] = (
            1.0 if board0.has_kingside_castling_rights(chess.BLACK) else 0.0
        )
        planes[b_idx, 116, :, :] = (
            1.0 if board0.has_queenside_castling_rights(chess.BLACK) else 0.0
        )

        planes[b_idx, 117, :, :] = board0.fullmove_number / 100.0
        planes[b_idx, 118, :, :] = board0.halfmove_clock / 100.0

    return torch.tensor(planes)


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


def _flat_index_of_move(move: chess.Move) -> int:
    """
    Return the flattened index (0-based, 0‥4671) that the policy head uses
    for `move`.  This lets us look the probability up in a *single* tensor
    instead of touching three indices.
    """
    plane = _get_move_type_index(move)  # 0‥72
    row = 7 - (move.from_square // 8)  # 0‥7  (network is flipped)
    col = move.from_square % 8  # 0‥7
    return plane * 64 + row * 8 + col  # 73 · 8 · 8 = 4672


def _decode_move(policies, fens, apply_softmax: bool = True):
    """
    Convert a batch of raw network outputs `policies` (Tensor) into
    *legal* UCI moves by picking the legal move with the highest predicted
    probability.

    • `policies[i]` is (73,8,8) and may still reside on the GPU.
    • If `apply_softmax` is True we turn logits → probabilities first;
      if you already trained with a soft-max head you can leave it False.
    """
    best_moves = []
    for logits, fen in zip(policies, fens):
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:  # stalemate / checkmate
            best_moves.append(None)
            continue

        # Vectorise: flatten once, option-ally soft-max once
        flat = logits.view(-1)  # shape (4672,)
        if apply_softmax:
            flat = torch.softmax(flat, dim=0)

        # Gather probabilities of *only* the legal moves
        idxs = torch.tensor(
            [_flat_index_of_move(m) for m in legal_moves], device=flat.device
        )
        probs = flat[idxs]  # shape (L,)

        best_moves.append(legal_moves[int(torch.argmax(probs))])

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
