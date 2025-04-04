import json
import re
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.chess_utils import column_letter_to_num, is_fen_valid, is_uci_valid


class AegisDataset(Dataset):
    def __init__(self, dir="datasets/aegis/data", raw_dir="datasets/aegis/raw_data"):
        self.raw_dir = Path(raw_dir)
        self.dir = Path(dir)
        self.files = list(self.dir.glob("*.jsonl"))  # Collect all JSONL files

        # Pre-load all data into memory during initialization
        self.data = []
        for file_path in self.files:
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        fen = list(data.keys())[0]
        history = data[fen]["history"]
        best_move = data[fen]["best_move"]
        x = self.encode_input([fen], [history])
        y = self.encode_output([best_move])
        return x[0], y[0]

    def encode_input(self, fens, histories):
        """Encode a batch of FEN strings into a batch of input tensors with shape (batch_size, 9, 8, 8)."""
        batch_size = len(fens)
        inputs = np.zeros((batch_size, 52, 8, 8), dtype=np.float32)

        types = ["p", "n", "b", "r", "q", "k"]
        for i, (fen, history) in enumerate(zip(fens, histories)):
            assert is_fen_valid(fen)
            board = chess.Board(fen)
            layer_idx = 0

            for fen in [fen] + history:
                # Encode board position for each piece type
                for type in types:
                    s = str(board)
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
                castling_tensor = np.zeros((8, 8), dtype=np.int8)
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
            inputs[i, layer_idx, :, :] = np.full(
                (8, 8), repetition_count, dtype=np.int8
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
            inputs[i, layer_idx, :, :] = np.ones((8, 8), dtype=np.int8)
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

    def decode_input(self, tensor):
        """Decode a batch of input tensors with shape (batch_size, 9, 8, 8) back into FEN strings and histories."""
        batch_size = tensor.shape[0]
        fens = []
        histories = []

        types = ["p", "n", "b", "r", "q", "k"]

        for i in range(batch_size):
            # Get the input for this batch item
            input_data = tensor[i].numpy()
            layer_idx = 0

            # Initialize variables to store current position and history
            current_fen = None
            history_fens = []

            # Process each position in the history (current + past positions)
            while layer_idx < input_data.shape[0]:
                board = chess.Board(None)  # Create empty board
                board.clear()

                # Process piece layers
                piece_map = {}
                for piece_type in types:
                    layer = input_data[layer_idx]
                    for rank in range(8):
                        for file in range(8):
                            val = layer[
                                7 - rank, file
                            ]  # Convert to chess board coordinates
                            if val == 1:  # White piece
                                piece = piece_type.upper()
                                square = chess.square(file, rank)
                                piece_map[square] = piece
                            elif val == -1:  # Black piece
                                piece = piece_type
                                square = chess.square(file, rank)
                                piece_map[square] = piece
                    layer_idx += 1

                # Set pieces on board
                for square, piece in piece_map.items():
                    board.set_piece_at(square, chess.Piece.from_symbol(piece))

                # Process castling rights
                castling_layer = input_data[layer_idx]
                layer_idx += 1

                # White kingside
                if castling_layer[7, 7] == 1:
                    board.castling_rights |= chess.BB_H1
                # White queenside
                if castling_layer[7, 0] == 1:
                    board.castling_rights |= chess.BB_A1
                # Black kingside
                if castling_layer[0, 7] == -1:
                    board.castling_rights |= chess.BB_H8
                # Black queenside
                if castling_layer[0, 0] == -1:
                    board.castling_rights |= chess.BB_A8

                # Process turn (last layer)
                if layer_idx < input_data.shape[0]:
                    turn_layer = input_data[layer_idx]
                    if turn_layer[0, 0] == 1:  # White to move
                        board.turn = chess.WHITE
                    else:  # Black to move
                        board.turn = chess.BLACK
                    layer_idx += 1

                # Get FEN for this position
                fen = board.fen()

                if current_fen is None:
                    current_fen = fen
                else:
                    history_fens.append(fen)

            fens.append(current_fen)
            histories.append(history_fens)

        return fens, histories

    def decode_output(self, tensor):
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
