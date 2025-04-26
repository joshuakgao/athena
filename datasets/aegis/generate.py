import json
import os
import sys
from pathlib import Path
import chess
import chess.pgn
import psutil
from tqdm import tqdm
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.chess_utils import is_fen_valid
from utils.logger import logger

raw_dir = "datasets/aegis/raw_data"
dir = "datasets/aegis/data"


def write_data_to_file(data, output_dir, file_index):
    """Write data to JSONL file and clear memory."""
    output_path = output_dir / f"shard_{file_index:04d}.jsonl"
    with open(output_path, "w") as f:
        for fen, move_data in data.items():
            json.dump({fen: move_data["best_move"]}, f)
            f.write("\n")
    logger.info(f"Created {output_path} with {len(data)} positions")
    return file_index + 1


def extract_eval_and_depth(comment):
    """Extract evaluation and depth from PGN comment."""
    match = re.search(r"([+-]?\d+(\.\d+)?)/(\d+)", comment)
    if match:
        eval_str = match.group(1)
        depth = int(match.group(3))
        eval_cp = int(float(eval_str) * 100)
        return eval_cp, depth
    return None, None


def process_game(game, data):
    """Process a single game and update the data dictionary."""
    board = game.board()

    for node in game.mainline():
        fen = board.fen()
        move = node.move

        if move not in board.legal_moves:
            logger.warning(f"Illegal move: {move} in FEN: {fen}")
            continue

        board.push(move)
        next_move_uci = move.uci()
        comment = node.comment
        depth = extract_eval_and_depth(comment)[1] or 0

        if fen not in data or depth > data[fen].get("depth", 0):
            data[fen] = {"best_move": next_move_uci, "depth": depth}


def generate():
    """Process PGN files and create JSONL shards."""
    data = {}
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_index = 0

    pgn_files = sorted(
        Path(raw_dir).glob("*.pgn"),
        key=lambda x: (
            -float(re.search(r"([\d.]+)s", x.name).group(1))
            if re.search(r"([\d.]+)s", x.name)
            else 0
        ),
    )

    for pgn_path in pgn_files:
        file_size = pgn_path.stat().st_size
        logger.info(f"Processing {pgn_path.name} ({file_size/1024/1024:.2f} MB)")

        game_counter = 0  # Initialize a counter for games

        with open(pgn_path, "r") as pgn_file:
            with tqdm(
                total=file_size, unit="B", unit_scale=True, desc=pgn_path.name
            ) as pbar:
                while True:
                    start_pos = pgn_file.tell()
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    process_game(game, data)
                    pbar.update(pgn_file.tell() - start_pos)

                    game_counter += 1  # Increment the game counter

                    # Log memory usage every 1000 games
                    if game_counter % 10000 == 0:
                        memory = psutil.virtual_memory()
                        logger.info(
                            f"Processed {game_counter} games. Memory usage: {memory.percent}%"
                        )

                    # Check memory usage and write data if memory is above 90%
                    memory = psutil.virtual_memory()
                    if memory.percent > 95:
                        logger.warning(
                            f"Memory usage is high ({memory.percent}%). Writing data to disk..."
                        )
                        file_index = write_data_to_file(data, output_dir, file_index)
                        data.clear()  # Clear the data dictionary to free memory

        # Log memory usage after processing the file
        memory = psutil.virtual_memory()
        logger.info(f"Memory usage after processing {pgn_path.name}: {memory.percent}%")

    # Write remaining data
    if data:
        file_index = write_data_to_file(data, output_dir, file_index)

    logger.info("Processing complete")


if __name__ == "__main__":
    generate()
