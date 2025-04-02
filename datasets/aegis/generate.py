import json
import shutil
import tempfile
from collections import defaultdict, deque
from pathlib import Path

import chess
import chess.pgn

from utils.chess_utils import is_fen_valid
from utils.logger import logger

raw_dir = "datasets/aegis/raw_data"
dir = "datasets/aegis/data"


def generate():
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp())

    # --- FIRST PASS: Write all FENs to temp files ---
    temp_file_index = 0
    temp_line_count = 0
    temp_jsonl = open(temp_dir / f"temp_{temp_file_index}.jsonl", "w")

    for pgn_path in Path(raw_dir).glob("*.pgn"):
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

                    if elo > 3000:
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

    # --- SECOND PASS: Deduplicate and keep highest ELO ---
    fen_entries = defaultdict(dict)  # {fen: {"best_move": ..., "elo": ..., ...}}

    for temp_file in temp_dir.glob("temp_*.jsonl"):
        with open(temp_file) as f:
            for line in f:
                entry = json.loads(line)
                fen = entry["fen"]

                # If this FEN is new or has higher ELO, keep it
                if fen not in fen_entries or entry["elo"] > fen_entries[fen]["elo"]:
                    fen_entries[fen] = {
                        "best_move": entry["best_move"],
                        "elo": entry["elo"],
                        "history": entry["history"],
                        "bot": entry["bot"],
                    }

    # --- Write final deduplicated JSONL files ---
    file_index = 0
    line_count = 0
    jsonl_file = open(output_dir / f"shard_{file_index}.jsonl", "w")

    for fen, data in fen_entries.items():
        jsonl_file.write(json.dumps({fen: data}) + "\n")
        line_count += 1

        if line_count >= 1_000_000:
            jsonl_file.close()
            file_index += 1
            line_count = 0
            jsonl_file = open(output_dir / f"shard_{file_index}.jsonl", "w")

    jsonl_file.close()
    shutil.rmtree(temp_dir)  # Clean up temp files
    logger.info("Finished processing all PGN files.")


generate()
