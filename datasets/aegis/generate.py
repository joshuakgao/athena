import json
import os
import shutil
import sys
from collections import deque
from pathlib import Path

import chess
import chess.pgn
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.chess_utils import is_fen_valid
from utils.logger import logger

raw_dir = "datasets/aegis/raw_data"
dir = "datasets/aegis/data"


def generate():
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary directory for intermediate files
    temp_dir = Path("datasets/aegis/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # --- FIRST PASS: Write all FENs to temp files ---
    temp_files = list(temp_dir.glob("temp_*.jsonl"))
    if temp_files:
        logger.info("Temporary files already exist. Skipping first pass.")
    else:
        temp_file_index = 0
        temp_line_count = 0
        temp_jsonl = open(temp_dir / f"temp_{temp_file_index}.jsonl", "w")

        pgn_files = list(Path(raw_dir).glob("*.pgn"))
        for pgn_path in tqdm(pgn_files):
            logger.info(f"Processing {pgn_path.name}...")

            with open(pgn_path) as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    white_elo = int(game.headers.get("WhiteElo", 3404))
                    black_elo = int(game.headers.get("BlackElo", 3404))
                    white_bot = game.headers.get("White", "Unknown")
                    black_bot = game.headers.get("Black", "Unknown")
                    board = game.board()
                    last_seven_fens = deque(maxlen=7)

                    for move in game.mainline_moves():
                        fen = board.fen()
                        board.push(move)

                        if not is_fen_valid(fen):
                            continue

                        next_move_uci = move.uci()
                        active_color = fen.split()[1]
                        elo = white_elo if active_color == "w" else black_elo
                        bot = white_bot if active_color == "w" else black_bot

                        if elo > 2500:
                            entry = {
                                "fen": fen,
                                "best_move": next_move_uci,
                                "elo": elo,
                                "history": list(last_seven_fens),
                                "bot": bot,
                            }
                            temp_jsonl.write(json.dumps(entry) + "\n")
                            temp_line_count += 1

                            if temp_line_count >= 1_000_000:
                                temp_jsonl.close()
                                temp_file_index += 1
                                temp_line_count = 0
                                temp_jsonl = open(
                                    temp_dir / f"temp_{temp_file_index}.jsonl", "w"
                                )

                        last_seven_fens.appendleft(fen)

        temp_jsonl.close()

    # --- SECOND PASS: Sort and Merge for Deduplication ---
    temp_files = list(temp_dir.glob("temp_*.jsonl"))
    logger.info(f"Number of temporary files: {len(temp_files)}")

    def sort_temp_file(input_path: Path, output_path: Path):
        logger.info(f"Sorting {input_path.name}...")
        lines = []
        with open(input_path, "r") as f:
            for line in f:
                lines.append(json.loads(line))
        lines.sort(key=lambda x: x["fen"])
        with open(output_path, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

    sorted_temp_files = []
    for temp_file in tqdm(temp_files, desc="Sorting temporary files"):
        sorted_file = temp_dir / f"sorted_{temp_file.name}"
        sort_temp_file(temp_file, sorted_file)
        sorted_temp_files.append(sorted_file)

    logger.info("Merging and deduplicating sorted files...")
    file_index = 0
    line_count = 0
    output_filename = output_dir / f"shard_{file_index}.jsonl"
    output_file = open(output_filename, "w")

    file_handles = [open(f, "r") for f in sorted_temp_files]
    current_lines = [
        json.loads(f.readline()) if (line := f.readline()) else None
        for f in file_handles
    ]

    while any(line is not None for line in current_lines):
        valid_entries = [
            (i, entry) for i, entry in enumerate(current_lines) if entry is not None
        ]
        if not valid_entries:
            break

        min_fen = min(entry[1]["fen"] for entry in valid_entries)
        best_entry = None

        for i, entry in valid_entries:
            if entry["fen"] == min_fen:
                if best_entry is None or entry["elo"] > best_entry["elo"]:
                    best_entry = entry
                # Read the next line from the file we just processed
                next_line = file_handles[i].readline()
                current_lines[i] = json.loads(next_line) if next_line else None

        if best_entry:
            output_file.write(json.dumps(best_entry) + "\n")
            line_count += 1
            if line_count >= 1_000_000:
                output_file.close()
                file_index += 1
                output_filename = output_dir / f"shard_{file_index}.jsonl"
                output_file = open(output_filename, "w")
                line_count = 0

    # Close all file handles
    for f in file_handles:
        f.close()
    output_file.close()

    # Clean up temporary files
    for sorted_file in sorted_temp_files:
        sorted_file.unlink()
    shutil.rmtree(temp_dir)
    logger.info("Finished processing all PGN files.")


generate()
