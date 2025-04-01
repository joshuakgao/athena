import json
from collections import deque  # Add at the top of your file
from pathlib import Path

import chess
import chess.pgn

from utils.chess_utils import is_fen_valid
from utils.logger import logger

raw_dir = "datasets/aegis/raw_data"
dir = "datasets/aegis/data"


def generate():
    """
    Process all PGN files in the directory and create a separate JSONL file for each PGN.
    """
    data = {}

    for pgn_path in Path(raw_dir).glob("*.pgn"):
        logger.info(f"Processing {pgn_path.name}...")

        with open(pgn_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  # End of file

                # Default set to Lc0 0.27.0 elo
                white_elo = int(game.headers.get("WhiteElo", 3404))
                black_elo = int(game.headers.get("BlackElo", 3404))
                white_bot = game.headers.get("White", "Unknown")
                black_bot = game.headers.get("Black", "Unknown")
                board = game.board()

                # Use a deque as a reverse queue (newest FEN first)
                last_seven_fens = deque(
                    maxlen=7
                )  # Automatically discards oldest when full

                for move in game.mainline_moves():
                    # Get the FEN before the move is made
                    fen = board.fen()
                    board.push(move)  # Make the move

                    if not is_fen_valid(fen):
                        continue

                    next_move_uci = move.uci()
                    active_color = fen.split()[1]  # 'w' or 'b'
                    elo = white_elo if active_color == "w" else black_elo
                    bot = white_bot if active_color == "w" else black_bot

                    if fen not in data or elo > data[fen]["elo"]:
                        data[fen] = {
                            "best_move": next_move_uci,
                            "elo": elo,
                            "history": list(last_seven_fens),  # Already newest-first
                            "bot": bot,
                        }

                    # Append new FEN to the front (reverse order)
                    last_seven_fens.appendleft(fen)

    # Write the best move for each FEN to multiple JSONL files, each with up to one million lines
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_index = 0
    line_count = 0
    jsonl_file = open(output_dir / f"shard_{file_index}.jsonl", "w")

    for fen, move_data in data.items():
        jsonl_file.write(json.dumps({fen: move_data}) + "\n")
        line_count += 1

        # If the current file reaches one million lines, start a new file
        if line_count >= 1_000_000:
            logger.info(f"Created {jsonl_file.name}.")
            jsonl_file.close()
            file_index += 1
            line_count = 0
            jsonl_file = open(output_dir / f"shard_{file_index}.jsonl", "w")

    # Close the last file
    jsonl_file.close()
    logger.info("Finished processing all PGN files.")


generate()
