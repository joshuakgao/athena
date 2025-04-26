import json
import re
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.chess_utils import column_letter_to_num, is_fen_valid, is_uci_valid
from utils.logger import logger


class AegisDataset(Dataset):
    def __init__(self, dir="datasets/aegis/data", raw_dir="datasets/aegis/raw_data"):
        self.raw_dir = Path(raw_dir)
        self.dir = Path(dir)
        self.files = list(self.dir.glob("*.jsonl"))  # Collect all JSONL files

        # Pre-load all data into memory during initialization
        self.data = []
        for file_path in self.files:
            with open(file_path, "r") as f:
                for line in tqdm(f):
                    data = json.loads(line.strip())
                    fen = list(data.keys())[0]
                    move = data[fen]
                    self.data.append((fen, move))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = self.encode_input([x])
        y = self.encode_output([y])
        return x[0], y[0]

    def encode_input(self, fens):
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

    def generate(self):
        """
        Process all PGN files in the directory and create a separate JSONL file for each PGN.
        """
        seen_fens = set()
        data = {}

        # Process each PGN file in the directory
        for pgn_path in Path(self.raw_dir).glob("*.pgn"):
            logger.info(f"Processing {pgn_path.name}...")

            with open(pgn_path) as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break  # End of file

                    board = game.board()

                    for move in game.mainline_moves():
                        # Get the FEN before the move is made
                        fen = board.fen()
                        # Make the move on the board
                        board.push(move)

                        if not is_fen_valid(fen):
                            continue

                        # Get the next move in UCI format
                        next_move_uci = move.uci()

                        if fen not in seen_fens:
                            data[fen] = {next_move_uci: 1}
                            seen_fens.add(fen)
                        else:
                            if next_move_uci not in data[fen]:
                                data[fen][next_move_uci] = 1
                            else:
                                data[fen][next_move_uci] += 1

        # Write the most used move for each FEN to multiple JSONL files, each with up to one million lines
        output_dir = self.dir
        output_dir.mkdir(parents=True, exist_ok=True)

        file_index = 0
        line_count = 0
        jsonl_file = open(output_dir / f"shard_{file_index}.jsonl", "w")

        for fen, moves in data.items():
            # Find the move with the highest count
            most_used_move = max(moves, key=moves.get)
            jsonl_file.write(json.dumps({fen: most_used_move}) + "\n")
            line_count += 1

            # If the current file reaches one million lines, start a new file
            if line_count >= 1_000_000:
                logger.info(f"Created {jsonl_file.name}.")
                jsonl_file.close()
                file_index += 1
                line_count = 0
                jsonl_file = open(output_dir / f"data_{file_index}.jsonl", "w")

        # Close the last file
        jsonl_file.close()

    def clean(self):
        """
        Remove all duplicate FENs across all JSONL files in the dataset directory.
        Maintains only the first occurrence of each FEN across all files.
        """
        logger.info(
            "Cleaning all files to remove duplicate FENs across entire dataset..."
        )

        # First pass: collect all data and identify duplicates
        all_data = []
        seen_fens = set()
        total_positions = 0
        duplicates_removed = 0

        for jsonl_path in self.dir.glob("*.jsonl"):
            with open(jsonl_path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    fen = list(data.keys())[0]

                    if fen not in seen_fens:
                        seen_fens.add(fen)
                        all_data.append((jsonl_path, data))
                        total_positions += 1
                    else:
                        duplicates_removed += 1

        logger.info(
            f"Found {total_positions} unique positions, removed {duplicates_removed} duplicates overall"
        )

        # Second pass: reorganize data by original files (without duplicates)
        file_contents = {path: [] for path in self.dir.glob("*.jsonl")}

        for path, data in all_data:
            file_contents[path].append(data)

        # Third pass: write cleaned data back to files
        for path, contents in file_contents.items():
            with open(path, "w") as f:
                for record in contents:
                    f.write(json.dumps(record) + "\n")

            logger.info(f"Wrote {len(contents)} unique positions to {path.name}")

        logger.info("Finished cleaning all files")


if __name__ == "__main__":
    aegis = AegisDataset()
    aegis.generate()
