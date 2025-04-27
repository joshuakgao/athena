import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import random


class AegisDataset(Dataset):
    def __init__(
        self, dir="datasets/aegis/data", n=10000000, no_load=False, test=False
    ):
        """
        Args:
            dir: Directory containing the dataset
            n: Number of samples to load (ignored for test set)
            no_load: If True, skip loading data
            test: If True, load the test set (test.parquet)
        """
        self.dir = Path(dir)
        self.test = test
        self.n = n
        self.metadata_path = self.dir / "metadata.json"
        self.data = pd.DataFrame(columns=["fen", "top_moves", "evals"])

        if not test:
            # Get all shards except test.parquet
            self.shard_paths = [
                p for p in self.dir.glob("*.parquet") if p.name != "test.parquet"
            ]
        else:
            # Use only the test.parquet file
            self.test_path = self.dir / "test.parquet"

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        if not no_load:
            self.sample_dataset()

    def sample_dataset(self):
        if self.test:
            # Load the entire test set
            self.data = pd.read_parquet(
                self.test_path, columns=["fen", "top_moves", "evals"]
            )
        else:
            # Load a sampled subset of the training set
            self.data = pd.DataFrame(columns=["fen", "top_moves", "evals"])
            num_shards = len(self.shard_paths)
            samples_per_shard = self.n // num_shards

            pieces = []
            for shard_path in tqdm(
                self.shard_paths, desc=f"Sampling ~{self.n} of Aegis"
            ):
                df = pd.read_parquet(shard_path, columns=["fen", "top_moves", "evals"])
                num_rows = df.shape[0]
                random_indexes = random.sample(
                    range(num_rows), min(samples_per_shard, num_rows)
                )
                sampled_df = df.iloc[random_indexes]
                pieces.append(sampled_df)

            self.data = pd.concat(pieces, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row["fen"]
        top_moves = row["top_moves"]
        evals = row["evals"]

        X = self.encode_position([fen])
        Y = self.encode_move([top_moves], [evals])

        return X[0], Y[0], fen, list(top_moves), list(evals)

    def encode_position(self, fens):
        """
        Simplified encoding → Tensor  (B, 119, 8, 8)

        Only encodes the current board state (no history).

        Piece planes (0-11):
            • 6 planes: white {P,N,B,R,Q,K}
            • 6 planes: black {p,n,b,r,q,k}

        Repetition planes (12-13):
            • plane-12 = 1 if position seen once before
            • plane-13 = 1 if position seen ≥2× before

        Static context (14-20):
            14 : side-to-move (1 if White, else 0)
            15 : white K-side castling (1 if available, else 0)
            16 : white Q-side castling
            17 : black K-side castling
            18 : black Q-side castling
            19 : full-move number / 100 (float)
            20 : half-move clock / 100 (float)
        """
        B = len(fens)
        planes = np.zeros((B, 21, 8, 8), dtype=np.float32)  # Now only 21 planes needed

        piece_symbols = "PNBRQKpnbrqk"  # order is important

        for b_idx, fen in enumerate(fens):
            board = chess.Board(fen)

            # --- 12 piece-presence planes (0-11) --------------------------------
            s = str(board).replace("\n", " ")
            squares = np.array(s.split(), dtype="<U1").reshape(8, 8)

            for p_idx, symbol in enumerate(piece_symbols):
                mask = (squares == symbol).astype(np.float32)
                planes[b_idx, p_idx, :, :] = mask

            # --- Repetition planes (12-13) --------------------------------------
            if board.is_repetition(1):
                planes[b_idx, 12, :, :] = 1
            if board.is_repetition(2):
                planes[b_idx, 13, :, :] = 1

            # --- Static context (14-20) -----------------------------------------
            planes[b_idx, 14, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

            planes[b_idx, 15, :, :] = (
                1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
            )
            planes[b_idx, 16, :, :] = (
                1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
            )
            planes[b_idx, 17, :, :] = (
                1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
            )
            planes[b_idx, 18, :, :] = (
                1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
            )

            planes[b_idx, 19, :, :] = board.fullmove_number / 100.0
            planes[b_idx, 20, :, :] = board.halfmove_clock / 100.0

        return torch.tensor(planes)

    def encode_move(self, top_moves: list, evals: list):
        """
        Encode moves into a probability distribution tensor weighted by centipawn scores.

        Args:
            top_moves: List of lists, where each inner list contains UCI moves
                        for a position, e.g., [["e2e4", "d2d4"], ["g1f3", "c2c4"], ...].
            evals: List of lists, where each inner list contains centipawn scores
                    corresponding to the moves in `top_moves`, e.g., [[25, 20], [30, 15], ...].

        Returns:
            Tensor of shape [batch_size, 4672] with centipawn-weighted probabilities.
        """
        batch_size = len(top_moves)
        outputs = torch.zeros((batch_size, 4672), dtype=torch.float32)

        for i, (moves, scores) in enumerate(zip(top_moves, evals)):
            if len(moves) == 0:
                continue  # No valid moves (unlikely in practice)

            # --- Convert centipawns to probabilities ---
            scores = np.array(scores, dtype=np.float32)

            # Option 3: Centipawn-to-win-probability (Stockfish-style)
            k = 0.003  # Tune this hyperparameter
            probs = 1 / (1 + np.exp(-k * scores))
            probs = probs / probs.sum()  # Normalize

            # Convert probs to a PyTorch tensor
            probs = torch.tensor(probs, dtype=torch.float32)

            # --- Assign probabilities to output tensor ---
            for move_uci, prob in zip(moves, probs):
                move = chess.Move.from_uci(move_uci)
                flat_idx = self.flat_index_of_move(move)
                outputs[i, flat_idx] = prob

        return outputs

    def get_move_type_index(self, move: chess.Move) -> int:
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

    def flat_index_of_move(self, move: chess.Move) -> int:
        """
        Return the flattened index (0-based, 0‥4671) that the policy head uses
        for `move`.  This lets us look the probability up in a *single* tensor
        instead of touching three indices.
        """
        plane = self.get_move_type_index(move)  # 0‥72
        row = 7 - (move.from_square // 8)  # 0‥7  (network is flipped)
        col = move.from_square % 8  # 0‥7
        return plane * 64 + row * 8 + col  # 73 · 8 · 8 = 4672

    def decode_move(
        self,
        policies,
        fens,
        apply_softmax: bool = True,
        repetition_penalty: float = 0.9,
    ):
        """
        Convert a batch of raw network outputs `policies` into legal UCI moves,
        returning both the best move and second-best (ponder) move for each position.

        Args:
            policies: Tensor of shape (batch_size, 73, 8, 8) - network policy outputs
            fens: List of FEN strings for each position in the batch
            apply_softmax: Whether to apply softmax to convert logits to probabilities
            repetition_penalty: Factor to penalize moves that lead to repetition (1.0 = no penalty)

        Returns:
            List of tuples: (best_move, ponder_move) for each position
                        (None, None) if no legal moves
        """
        best_moves = []
        for logits, fen in zip(policies, fens):
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if not legal_moves:  # stalemate / checkmate
                best_moves.append((None, None))
                continue

            # Vectorise: flatten once, optionally soft-max once
            flat = logits.view(-1)  # shape (4672,)
            if apply_softmax:
                flat = torch.softmax(flat, dim=0)

            # Get indices and original probabilities for legal moves
            idxs = torch.tensor(
                [self.flat_index_of_move(m) for m in legal_moves], device=flat.device
            )
            probs = flat[idxs].clone()  # shape (L,)

            # Apply repetition penalty
            if repetition_penalty < 1.0:
                for i, move in enumerate(legal_moves):
                    board.push(move)
                    if board.is_repetition(2):  # Would this move cause repetition?
                        probs[i] *= repetition_penalty
                    board.pop()

            # Get top moves - handle cases with only 1 legal move
            k = min(2, len(legal_moves))
            top_indices = torch.topk(probs, k=k).indices

            best_move = legal_moves[int(top_indices[0])]
            ponder_move = legal_moves[int(top_indices[1])] if k > 1 else None

            best_moves.append((best_move, ponder_move))

        return best_moves
