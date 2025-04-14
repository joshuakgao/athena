import bisect
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.chess_utils import column_letter_to_num, is_uci_valid
from utils.logger import logger


class AegisDataset(Dataset):
    def __init__(self, dir="datasets/aegis/data", max_cache_size=5):
        """
        Args:
            dir: Directory containing the dataset
            max_cache_size: Maximum number of parquet files to keep in memory at once
        """
        self.dir = Path(dir)
        self.files = sorted(self.dir.glob("*.parquet"))
        self.metadata_path = self.dir / "metadata.json"

        # LRU cache for loaded files
        self.max_cache_size = max(max_cache_size, 1)  # ensure at least 1
        self.file_cache = OrderedDict()  # Maintains access order

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Store file information instead of full index map
        self.file_info = []
        known_rows_per_file = 10_000_000
        last_file_index = len(self.files) - 1

        # Pre-calculate file offsets and row counts
        for file_idx in tqdm(range(len(self.files))):
            if file_idx < last_file_index:
                num_rows = known_rows_per_file
            else:
                # Only load last file when needed
                last_file_path = self.files[last_file_index]
                parquet_file = pd.read_parquet(last_file_path, engine="pyarrow")
                num_rows = len(parquet_file)
            self.file_info.append((file_idx, num_rows))

        # Calculate cumulative offsets
        self.cumulative_offsets = [0]
        for _, num_rows in self.file_info:
            self.cumulative_offsets.append(self.cumulative_offsets[-1] + num_rows)

    def _load_file(self, file_idx):
        """Lazy load a file with LRU caching.

        Args:
            file_idx: Index of the file to load

        Returns:
            None (loads file into cache)
        """
        # Check if file is already in cache
        if file_idx in self.file_cache:
            # Move to end to mark as recently used
            self.file_cache.move_to_end(file_idx)
            return

        # Load the file from disk
        file_path = self.files[file_idx]

        # Read parquet file efficiently using pyarrow
        try:
            df = pd.read_parquet(file_path, engine="pyarrow")
            records = df.to_dict("records")

            # Free memory immediately after conversion
            del df

            # Add to cache with LRU management
            self.file_cache[file_idx] = records
            self.file_cache.move_to_end(file_idx)

            # Enforce cache size limit
            if len(self.file_cache) > self.max_cache_size:
                # Remove least recently used file
                oldest_key = next(iter(self.file_cache))
                del self.file_cache[oldest_key]

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def _get_file_and_row(self, idx):
        """Calculate which file and row corresponds to the given index"""
        if idx >= len(self):
            raise IndexError("Index out of bounds")

        # Binary search to find the right file
        file_idx = bisect.bisect_right(self.cumulative_offsets, idx) - 1
        row_idx = idx - self.cumulative_offsets[file_idx]
        return file_idx, row_idx

    def __len__(self):
        return self.metadata["total_positions"]

    def __getitem__(self, idx):
        file_idx, row_idx = self._get_file_and_row(idx)
        self._load_file(file_idx)

        row = self.file_cache[file_idx][row_idx]
        fen = row["fen"]
        history = row["history"]
        best_move = row["best_move"]
        x = self.encode_input([fen], [history])
        y = self.encode_output([best_move])
        return x[0], y[0], fen, best_move

    def clear_cache(self):
        """Clear the file cache manually if needed"""
        self.file_cache.clear()

    def encode_input(self, fens, histories):
        """
        Encode a batch of FEN strings into a batch of input tensors with shape (batch_size, 69, 8, 8).
        8 layers for each piece type (pawn, knight, bishop, rook, queen, king),
        1 layer for castling rights, 1 layer for en passant square,
        8 fens are then encoded in this manner (1 current + 7 history),
        1 layer for repetition count, 1 layer for halfmove clock,
        1 layer for fullmove number, and 1 layer for turn.
        The last layer is a constant 1s tensor to indicate the board area to reduce the effect of edge blurring in the convlutional layers.
        """
        batch_size = len(fens)
        inputs = np.zeros((batch_size, 62, 8, 8), dtype=np.float32)

        types = ["p", "n", "b", "r", "q", "k"]
        for i, (fen, history) in enumerate(zip(fens, histories)):
            history = list(history)
            while len(history) < 7:
                history.append(None)

            board = chess.Board(fen, chess960=True)
            layer_idx = 0

            history.insert(0, fen)
            for _fen in history:
                _board = chess.Board(_fen, chess960=True)
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
                    board_mat = [
                        [int(x) for x in row.split(" ")] for row in s.split("\n")
                    ]
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

            # Encode en passant square
            ep_square = board.ep_square
            if ep_square is not None:
                ep_row, ep_col = chess.square_rank(ep_square), chess.square_file(
                    ep_square
                )
                turn = board.turn
                if turn == chess.WHITE:
                    inputs[i, layer_idx, ep_row, ep_col] = 1
                else:
                    inputs[i, layer_idx, ep_row, ep_col] = -1
            else:
                inputs[i, layer_idx, :, :] = np.zeros((8, 8), dtype=np.float32)
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
            inputs[i, layer_idx, :, :] = np.full(
                (8, 8), repetition_count, dtype=np.float32
            )
            layer_idx += 1

            # Encode number of halfmoves with no capture or pawn move
            halfmove_count = board.halfmove_clock
            inputs[i, layer_idx, :, :] = np.full(
                (8, 8), halfmove_count, dtype=np.float32
            )
            layer_idx += 1

            # Encode number of fullmoves
            fullmove_count = board.fullmove_number
            inputs[i, layer_idx, :, :] = np.full(
                (8, 8), fullmove_count, dtype=np.float32
            )
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

    def encode_output(self, ucis):
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

    def decode_output(self, policies, fens):
        """Decode policy output tensor (shape [2, 8, 8]) to a single UCI move."""
        best_moves = []
        for policy, fen in zip(policies, fens):
            board = chess.Board(fen, chess960=True)
            if not board.is_valid():
                board = chess.Board(fen, chess960=True)

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
                score = (
                    policy_from[from_sq[0], from_sq[1]] * policy_to[to_sq[0], to_sq[1]]
                )
                move_scores.append(score.item())

            # Select move with highest score
            best_move = legal_moves[np.argmax(move_scores)]
            best_moves.append(best_move)
        return best_moves
