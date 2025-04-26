import json
import os
import re
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.logger import logger

raw_dir = "datasets/aegis/raw_data"
dir = "datasets/aegis/data"
MAX_NUM_POSITIONS = 50_000_000


def write_data_to_file(data, output_dir):
    """Write data to JSONL shards"""
    shard_size = 1_000_000
    items = list(data.items())
    for i in range(0, len(items), shard_size):
        shard = items[i : i + shard_size]
        output_path = output_dir / f"aegis_{i // shard_size:04d}.jsonl"
        with open(output_path, "w") as f:
            for fen, move_data in shard:
                json.dump({fen: move_data}, f)
                f.write("\n")
        logger.info(f"Created {output_path} with {len(shard)} positions")


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
    data = {}
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pgn_files = sorted(
        Path(raw_dir).glob("*.pgn"),
        key=lambda x: (
            -float(re.search(r"([\d.]+)s", x.name).group(1))
            if re.search(r"([\d.]+)s", x.name)
            else 0
        ),
    )

    for pgn_path in pgn_files:
        with open(pgn_path, "r") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                process_game(game, data)

                if len(data) >= MAX_NUM_POSITIONS:
                    data = dict(list(data.items())[:MAX_NUM_POSITIONS])
                    write_data_to_file(data, output_dir)
                    return


if __name__ == "__main__":
    generate()
