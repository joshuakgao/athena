import json
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

    # Process each PGN file in the directory
    for pgn_path in Path(raw_dir).glob("*.pgn"):
        logger.info(f"Processing {pgn_path.name}...")

        with open(pgn_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  # End of file

                # default set to Lc0 0.27.0 elo
                white_elo = game.headers.get("WhiteElo", 3404)
                black_elo = game.headers.get("BlackElo", 3404)
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
                    active_color = fen.split()[1]  # Second field is the active color
                    elo = white_elo if active_color == "w" else black_elo

                    if fen not in data:
                        data[fen] = {"best_move": next_move_uci, "elo": elo}
                    else:
                        # Only update best move if it is made by a stronger engine
                        if elo > data[fen]["elo"]:
                            data[fen]["elo"] = elo
                            data[fen]["best_move"] = next_move_uci

    # Write the best move for each FEN to multiple JSONL files, each with up to one million lines
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_index = 0
    line_count = 0
    jsonl_file = open(output_dir / f"shard_{file_index}.jsonl", "w")

    for fen, move_data in data.items():
        jsonl_file.write(json.dumps({fen: move_data["best_move"]}) + "\n")
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
