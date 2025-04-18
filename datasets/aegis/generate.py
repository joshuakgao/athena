import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.logger import logger

# --- Configuration ---
raw_dir = "datasets/aegis/raw_data"
dir = "datasets/aegis/data"
temp_dir_path = Path("datasets/aegis/temp_parquet")
output_dir_path = Path(dir)
test_output_path = output_dir_path / "test.parquet"  # Path for test set
rows_per_temp_parquet_write = 1_000_000
positions_per_shard = 1_000_000
min_elo_threshold = 2500
min_depth_threshold = 0
samples_per_shard_for_test = 10  # Number of samples per shard for test set

# Calculate 90% of available memory
available_memory = psutil.virtual_memory().available
sort_memory_gb = max(1, int(available_memory * 0.9 / (1024**3)))
sort_memory = f"{sort_memory_gb}G"

# Use all available workers for sorting
sort_parallelism = psutil.cpu_count(logical=True)
sort_temp_dir = "datasets/aegis/sort_temp"
Path(sort_temp_dir).mkdir(parents=True, exist_ok=True)  # Ensure sort temp dir exists

# Define PyArrow schema for consistency
ARROW_SCHEMA = pa.schema(
    [
        ("fen", pa.string()),
        ("best_move", pa.string()),
        ("elo", pa.int32()),
        ("history", pa.list_(pa.string())),
        ("bot", pa.string()),
        ("depth", pa.int32()),
        ("eval", pa.float32()),
    ]
)


class RunningStats:
    """Class to calculate running statistics without storing all values."""

    def __init__(self):
        self.count = 0
        self.min = float("inf")
        self.max = float("-inf")
        self.sum = 0

    def update(self, value):
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0

    def to_dict(self):
        return {
            "count": self.count,
            "avg": round(self.avg, 2),
            "min": int(self.min) if self.count > 0 else None,
            "max": int(self.max) if self.count > 0 else None,
        }


def write_batch_to_parquet(batch, file_path, schema):
    """Helper function to write a batch of records to a Parquet file."""
    if not batch:
        logger.warning(f"Attempted to write an empty batch to {file_path}. Skipping.")
        return 0
    try:
        # Ensure history is list, not numpy array
        for record in batch:
            if "history" in record and isinstance(record["history"], np.ndarray):
                record["history"] = record["history"].tolist()

        table = pa.Table.from_pylist(batch, schema=schema)
        pq.write_table(table, file_path)
        written_count = len(batch)
        return written_count
    except pa.ArrowInvalid as e:
        logger.error(f"ArrowInvalid error writing batch to {file_path}: {e}")
        # Debug info
        logger.error(f"Schema: {schema}")
        if batch:
            logger.error(f"First item type: {type(batch[0])}")
            logger.error(f"First item keys: {batch[0].keys()}")
            for k, v in batch[0].items():
                logger.error(f"  Key '{k}', Type: {type(v)}")
                if isinstance(v, list) and v:
                    logger.error(f"    List element type: {type(v[0])}")
        raise
    except Exception as e:
        logger.error(f"Error writing batch to {file_path}: {e}")
        logger.error(f"Problematic batch (first 5 items): {batch[:5]}")
        raise


def sample_and_split_batch(batch, num_samples):
    """Samples items from batch for test set, returns train batch and test samples."""
    if not batch:
        return [], []

    actual_samples = min(num_samples, len(batch))
    if actual_samples == 0:
        return batch, []

    sample_indices = random.sample(range(len(batch)), actual_samples)
    test_samples = [batch[i] for i in sample_indices]

    train_batch = [item for i, item in enumerate(batch) if i not in sample_indices]
    return train_batch, test_samples


def normalize_fen(fen_string):
    """Normalizes a FEN string by removing en passant, halfmove, and fullmove counters."""
    parts = fen_string.split()
    if len(parts) == 6:
        parts[3] = "-"  # Remove en passant target
        parts[4] = "0"  # Reset halfmove clock
        parts[5] = "1"  # Reset fullmove number (or set to 1)
        return " ".join(parts)
    else:
        logger.warning(
            f"Encountered potentially invalid FEN for normalization: {fen_string}"
        )
        return fen_string


def extract_eval_and_depth(comment):
    """
    Extracts centipawn evaluation and depth from a PGN comment.
    Looks for pattern like {+0.28/4 0.17s}
    Returns: (eval_cp, depth)
    """
    match = re.search(r"([+-]?\d+(\.\d+)?)/(\d+)", comment)
    if match:
        eval_str = match.group(1)
        depth = int(match.group(3))
        eval_cp = int(float(eval_str) * 100)
        return eval_cp, depth
    return None, None


def parse_jsonl_into_parquet(
    jsonl_path, temp_dir_path, schema, rows_per_write=1_000_000
):
    """
    Reads each line of a JSONL file, e.g.:
      {
        "fen": "6k1/4Rppp/8/8/8/8/5PPP/6K1 w - -",
        "evals": [
          {
            "pvs": [
              {"mate":1, "line":"e7e8"},
              {"cp":100, "line":"e7e5 e5e6 ..."}
            ],
            "depth": 99
          },
          ...
        ]
      }

    For each eval_item in "evals":
      1. Identify the *first* PV's line as the "best move" source.
      2. For *each* PV in that eval_item:
         - parse the moves in 'line',
         - push up to 7 moves to get 7 subsequent FENs,
         - store that list in "history",
         - store "best_move" from (1) so all PVs share the same best_move for that eval_item,
         - set "eval" to a big number if 'mate' in the PV, else 'cp', else 0,
         - store "depth" from eval_item["depth"].

    Writes rows into "temp_jsonl_*.parquet" using your ARROW_SCHEMA.
    """
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    batch = []
    file_index = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        # get number of lines in jsonl file
        logger.info("Getting number of lines in jsonl file.")
        count = 0
        for line in f:
            count += 1

    with open(jsonl_path, "r", encoding="utf-8") as f:
        logger.info(f"Process {jsonl_path.name} file of length {count}.")
        for line_num, line in tqdm(
            enumerate(f, start=1), total=count, desc=f"Processing {jsonl_path.name}"
        ):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON at line {line_num}: {e}")
                continue

            # Grab FEN + evals
            fen = data.get("fen", "")
            fen = fen.split()
            fen.append("0")
            fen.append("1")
            root_fen = " ".join(fen)
            evals_list = data.get("evals", [])
            if not fen or not evals_list:
                logger.info("Fen or evals list not found")
                continue

            for eval_item in evals_list:
                pvs = eval_item.get("pvs", [])
                depth_val = eval_item.get("depth", 0)
                if not pvs:
                    logger.info("Pvs not found")
                    continue  # no PVs -> skip

                for pv in pvs:
                    best_line_str = pv.get("line", "")

                    if best_line_str:
                        # The very first token in the line is the best move
                        moves = best_line_str.split()
                        fen_history = deque(maxlen=7)
                        board = chess.Board(root_fen, chess960=True)
                        cp = pv.get("cp", "")
                        if not cp:
                            if board.turn == chess.WHITE:
                                cp = 9999
                            else:
                                cp = -9999
                        for i, move in enumerate(moves):
                            fen = board.fen()
                            fen = normalize_fen(fen)
                            best_move = move
                            entry = {
                                "fen": fen,
                                "best_move": best_move,
                                "elo": 3529,  # for stockfish
                                "history": list(fen_history),
                                "bot": "Stockfish",  # placeholder or set to something relevant
                                "depth": depth_val,
                                "eval": float(cp),
                            }
                            batch.append(entry)

                            fen_history.appendleft(fen)
                            try:
                                board.push_uci(best_move)
                            except:
                                logger.warning(
                                    f"Invalid move '{best_move}' encountered. Breaking out."
                                )
                                break

                            # Write out periodically
                            if len(batch) >= rows_per_write:
                                temp_file_path = (
                                    temp_dir_path / f"temp_jsonl_{file_index}.parquet"
                                )
                                written = write_batch_to_parquet(
                                    batch, temp_file_path, schema
                                )
                                print(f"Wrote {written} rows to {temp_file_path.name}")
                                batch.clear()
                                file_index += 1

    # Final flush
    if batch:
        temp_file_path = temp_dir_path / f"temp_jsonl_{file_index}.parquet"
        written = write_batch_to_parquet(batch, temp_file_path, schema)
        print(f"Wrote final {written} rows to {temp_file_path.name}")
        batch.clear()


def generate():
    output_dir_path.mkdir(parents=True, exist_ok=True)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    # --- Additional stats: wins/draws, depth tracking ---
    white_win_count = 0
    black_win_count = 0
    draw_count = 0

    # Running stats for ELO (existing)
    white_stats = RunningStats()
    black_stats = RunningStats()
    overall_stats = RunningStats()

    # Running stats for depth
    white_depth_stats = RunningStats()
    black_depth_stats = RunningStats()
    overall_depth_stats = RunningStats()

    # --- FIRST PASS: Write all FENs to temp Parquet files ---
    temp_files = sorted(list(temp_dir_path.glob("temp_*.parquet")))

    total_games = 0
    input_positions_estimate = 0

    if not temp_files:
        logger.info(
            "No temporary Parquet files found. Starting First Pass: Processing PGNs..."
        )
        temp_file_index = 0
        current_temp_batch = []
        total_written_to_temp = 0

        pgn_files = sorted(list(Path(raw_dir).glob("*.pgn")))
        if pgn_files:
            for pgn_path in tqdm(pgn_files, desc="Processing PGN files"):
                logger.info(f"Processing {pgn_path.name}...")
                games_in_file = 0
                positions_in_file_batch = 0

                with open(pgn_path, errors="replace", encoding="utf-8") as pgn_file:
                    while True:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break

                        total_games += 1
                        games_in_file += 1

                        # Parse basic info
                        white_bot = game.headers.get("White", "Unknown")
                        black_bot = game.headers.get("Black", "Unknown")
                        white_elo_str = game.headers.get("WhiteElo", "0")
                        black_elo_str = game.headers.get("BlackElo", "0")
                        starting_fen = game.headers.get("FEN", "")

                        # Assign ELO for known bots or parse
                        if white_bot == "Lc0":
                            white_elo = 3404
                        elif white_bot == "Stockfish 101217 64 BMI2":
                            white_elo = 3529
                        else:
                            white_elo = (
                                int(white_elo_str) if white_elo_str.strip() else 0
                            )

                        if black_bot == "Lc0":
                            black_elo = 3404
                        elif black_bot == "Stockfish 101217 64 BMI2":
                            black_elo = 3529
                        else:
                            black_elo = (
                                int(black_elo_str) if black_elo_str.strip() else 0
                            )

                        # If both sides below threshold, skip entire game
                        if (
                            white_elo < min_elo_threshold
                            and black_elo < min_elo_threshold
                        ):
                            continue

                        # Count result if the game is not skipped
                        result = game.headers.get("Result", "")
                        if result == "1-0":
                            white_win_count += 1
                        elif result == "0-1":
                            black_win_count += 1
                        elif result == "1/2-1/2":
                            draw_count += 1

                        if starting_fen:
                            starting_fen = normalize_fen(starting_fen)
                            board = chess.Board(starting_fen, chess960=True)
                            if not board.is_valid():
                                continue
                        else:
                            board = game.board()
                        last_seven_fens = deque(maxlen=7)

                        for node in game.mainline():
                            move = node.move
                            current_fen_unnormalized = board.fen()

                            # Validate move BEFORE pushing
                            if (
                                move.uci() not in ["e1g1", "e1c1", "e8g8", "e8c8"]
                                and move not in board.legal_moves
                            ):
                                print(starting_fen)
                                logger.warning(
                                    f"Illegal move {move.uci()} in position {board.fen()}"
                                )
                                break  # or continue to skip just this move

                            # Get move info before pushing
                            next_move_uci = move.uci()
                            comment = node.comment
                            eval_cp, depth = None, None
                            if comment:
                                eval_cp, depth = extract_eval_and_depth(comment)

                                if eval_cp is None or depth is None:
                                    eval_cp = 0
                                    depth = 0  # book move

                                eval_cp = min(max(eval_cp, -9999), 9999)
                            else:
                                board.push(
                                    move
                                )  # Still need to advance even if we skip
                                last_seven_fens.appendleft(
                                    normalize_fen(current_fen_unnormalized)
                                )
                                continue

                            norm_fen = normalize_fen(current_fen_unnormalized)
                            if (
                                norm_fen == current_fen_unnormalized
                                and len(current_fen_unnormalized.split()) != 6
                            ):
                                logger.warning(
                                    f"Skipping invalid FEN: {current_fen_unnormalized} in game {total_games}"
                                )
                                continue

                            # Only push the move once we've validated everything
                            board.push(move)

                            fen_parts = current_fen_unnormalized.split()
                            active_color = fen_parts[1] if len(fen_parts) > 1 else "w"
                            elo = white_elo if active_color == "w" else black_elo
                            bot = white_bot if active_color == "w" else black_bot
                            if (
                                elo >= min_elo_threshold
                                and depth >= min_depth_threshold
                            ):
                                entry = {
                                    "fen": norm_fen,
                                    "best_move": next_move_uci,
                                    "elo": elo,
                                    "history": list(last_seven_fens),
                                    "bot": bot,
                                    "depth": depth,
                                    "eval": eval_cp,
                                }
                                current_temp_batch.append(entry)
                                positions_in_file_batch += 1
                                input_positions_estimate += 1

                            last_seven_fens.appendleft(norm_fen)

                            if len(current_temp_batch) >= rows_per_temp_parquet_write:
                                temp_file_path = (
                                    temp_dir_path / f"temp_{temp_file_index}.parquet"
                                )
                                written = write_batch_to_parquet(
                                    current_temp_batch, temp_file_path, ARROW_SCHEMA
                                )
                                if written > 0:
                                    logger.info(
                                        f"Written temp batch to {temp_file_path.name} ({written} records)"
                                    )
                                    temp_file_index += 1
                                total_written_to_temp += written
                                current_temp_batch.clear()

                logger.info(
                    f"Finished {pgn_path.name}. Games: {games_in_file}. Positions added: {positions_in_file_batch}."
                )

            if current_temp_batch:
                temp_file_path = temp_dir_path / f"temp_{temp_file_index}.parquet"
                written = write_batch_to_parquet(
                    current_temp_batch, temp_file_path, ARROW_SCHEMA
                )
                if written > 0:
                    logger.info(
                        f"Written final temp batch to {temp_file_path.name} ({written} records)"
                    )
                total_written_to_temp += written
                current_temp_batch.clear()

            logger.info(f"TOTAL PGN GAMES PARSED: {total_games}")
            logger.info(
                f"Total records written to temp Parquet files: {total_written_to_temp:,}"
            )
            if total_written_to_temp != input_positions_estimate:
                logger.warning(
                    f"Mismatch: estimated positions ({input_positions_estimate:,}), "
                    f"written ({total_written_to_temp:,})."
                )

        # parse lichess jsonl dataset
        jsonl_files = list(Path(raw_dir).glob("*.jsonl"))
        for jfile in jsonl_files:
            logger.info(f"Parsing JSONL file: {jfile.name}")
            parse_jsonl_into_parquet(
                jsonl_path=jfile,
                temp_dir_path=temp_dir_path,
                schema=ARROW_SCHEMA,
                rows_per_write=500_000,  # or whatever chunk size
            )
    else:
        logger.info(
            f"Found {len(temp_files)} existing temp Parquet files, skipping PGN processing."
        )
        logger.info("Estimating input position count from existing temp files...")
        input_count_sort_est = 0
        for f in tqdm(temp_files, desc="Estimating size"):
            try:
                input_count_sort_est += pq.read_metadata(f).num_rows
            except Exception as e:
                logger.warning(f"Could not read metadata for {f.name}: {e}")
        logger.info(f"Estimated input positions for sort: {input_count_sort_est:,}")
        input_positions_estimate = input_count_sort_est

    # --- SECOND PASS: External Sort + Reduce ---
    temp_files = sorted(list(temp_dir_path.glob("temp_*.parquet")))
    logger.info("Starting Second Pass: Deduplication and Filtering via External Sort")
    process = psutil.Process()
    initial_mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Memory before sort pipe: {initial_mem_mb:.2f} MB")

    current_shard_batch = []
    shard_index = 0
    total_positions_train_final = 0
    input_count_sort_actual = 0

    test_set_accumulated = []
    total_positions_test_final = 0

    sort_command = [
        "sort",
        "-t",
        "\t",
        "-k1,1",  # Sort by FEN
        "-k2,2",  # Then by history string
        "-k3,3nr",  # Then by depth descending (numeric, reverse)
        "-S",
        sort_memory,
        f"--parallel={sort_parallelism}",
        "--stable",
    ]
    if sort_temp_dir:
        sort_command.extend(["-T", sort_temp_dir])

    logger.info(f"Using sort command: {' '.join(sort_command)}")

    sort_proc = None
    unique_lines_processed = 0

    try:
        sort_proc = subprocess.Popen(
            sort_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=8192,
        )

        logger.info("Feeding data to external sort from temp Parquet files...")
        feeding_start_time = time.time()

        try:
            for temp_file_path in tqdm(temp_files, desc="Feeding temp files"):
                try:
                    table = pq.read_table(temp_file_path)
                    df = table.to_pandas()

                    for _, row in df.iterrows():
                        try:
                            entry_dict = row.to_dict()

                            # Validate
                            if not all(
                                k in entry_dict
                                for k in [
                                    "fen",
                                    "best_move",
                                    "elo",
                                    "history",
                                    "bot",
                                    "depth",
                                    "eval",
                                ]
                            ):
                                logger.warning(
                                    f"Skipping row with missing keys in {temp_file_path}: {entry_dict}"
                                )
                                continue

                            history_value = entry_dict["history"]
                            if isinstance(history_value, np.ndarray):
                                entry_dict["history"] = history_value.tolist()
                            elif not isinstance(history_value, list):
                                entry_dict["history"] = []

                            entry_dict["history"] = [
                                str(h) for h in entry_dict["history"]
                            ]

                            fen = str(entry_dict["fen"])
                            history_list = entry_dict["history"]
                            history_str = "||".join(history_list)
                            elo = int(entry_dict["elo"])
                            best_move = str(entry_dict["best_move"])
                            bot = str(entry_dict["bot"])
                            depth = int(entry_dict["depth"])
                            eval_ = float(entry_dict["eval"])

                            clean_entry = {
                                "fen": fen,
                                "best_move": best_move,
                                "elo": elo,
                                "history": history_list,
                                "bot": bot,
                                "depth": depth,
                                "eval": eval_,
                            }
                            original_json_str = json.dumps(
                                clean_entry, ensure_ascii=False, separators=(",", ":")
                            )
                            tsv_line = (
                                f"{fen}\t{history_str}\t{depth}\t{original_json_str}\n"
                            )
                            sort_proc.stdin.write(tsv_line)
                            input_count_sort_actual += 1

                        except BrokenPipeError:
                            logger.error(
                                "Broken Pipe Error: Sort process terminated unexpectedly."
                            )
                            raise
                        except Exception as e:
                            logger.error(
                                f"Unexpected error writing to sort stdin for row {row}: {e}",
                                exc_info=True,
                            )
                            raise

                except Exception as read_e:
                    logger.error(
                        f"Failed to read or process {temp_file_path}: {read_e}",
                        exc_info=True,
                    )
                    continue

        finally:
            if sort_proc and sort_proc.stdin and not sort_proc.stdin.closed:
                try:
                    sort_proc.stdin.close()
                    logger.info("Closed sort stdin.")
                except OSError as e:
                    logger.warning(f"Could not close sort stdin: {e}")

        feeding_duration = time.time() - feeding_start_time
        logger.info(
            f"Finished feeding {input_count_sort_actual:,} lines to sort in {feeding_duration:.2f} seconds."
        )

        mem_after_feed_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(
            f"Memory after feeding, before reduction: {mem_after_feed_mb:.2f} MB"
        )

        # --- Process the sorted output ---
        current_key = None  # Dedup key: (fen, history_str)
        output_start_time = time.time()
        total_lines_est = (
            input_count_sort_actual
            if input_count_sort_actual > 0
            else input_positions_estimate
        )

        output_progress = tqdm(
            sort_proc.stdout,
            desc=f"Reducing sorted data (Shard {shard_index})",
            unit=" lines",
            total=total_lines_est,
            mininterval=2.0,
        )

        for sorted_line in output_progress:
            try:
                parts = sorted_line.strip().split("\t", 3)
                if len(parts) != 4:
                    logger.warning(
                        f"Skipping malformed sorted line (expected 4 fields): {sorted_line.strip()[:100]}..."
                    )
                    continue
                fen, history_str, depth_str, original_json_str = parts
                key = (fen, history_str)

            except Exception as e:
                logger.warning(
                    f"Skipping malformed sorted line (Error parsing fields): {sorted_line.strip()[:100]}... - {e}"
                )
                continue

            if key != current_key:
                unique_lines_processed += 1
                current_key = key
                try:
                    best_entry_dict = json.loads(original_json_str)
                    current_shard_batch.append(best_entry_dict)

                    # If batch full, write shard
                    if len(current_shard_batch) >= positions_per_shard:
                        logger.info(
                            f"Shard {shard_index} full ({len(current_shard_batch)} records). Sampling for test set..."
                        )
                        train_batch_part, test_samples = sample_and_split_batch(
                            current_shard_batch, samples_per_shard_for_test
                        )
                        if test_samples:
                            test_set_accumulated.extend(test_samples)
                            total_positions_test_final += len(test_samples)
                            logger.info(
                                f"Sampled {len(test_samples)} records for test set from shard {shard_index}."
                            )

                        shard_file_path = (
                            output_dir_path / f"shard_{shard_index}.parquet"
                        )
                        if train_batch_part:
                            written = write_batch_to_parquet(
                                train_batch_part, shard_file_path, ARROW_SCHEMA
                            )
                            logger.info(
                                f"Written training shard {shard_index} ({written} records)"
                            )
                            total_positions_train_final += written

                            # Update ELO + Depth stats for the final training positions
                            for entry in train_batch_part:
                                stat_elo = entry["elo"]
                                stat_depth = entry["depth"]
                                color = entry["fen"].split(" ")[1]
                                if color == "w":
                                    white_stats.update(stat_elo)
                                    white_depth_stats.update(stat_depth)
                                else:
                                    black_stats.update(stat_elo)
                                    black_depth_stats.update(stat_depth)
                                overall_stats.update(stat_elo)
                                overall_depth_stats.update(stat_depth)

                        else:
                            logger.warning(
                                f"Shard {shard_index} is empty after sampling."
                            )

                        current_shard_batch.clear()
                        shard_index += 1
                        output_progress.set_description(
                            f"Reducing sorted data (Shard {shard_index})"
                        )

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Could not parse JSON for unique key {current_key}: {e} - JSON: {original_json_str[:100]}..."
                    )
                except KeyError as e:
                    logger.warning(
                        f"Missing key processing unique key {current_key}: {e} - Dict: {best_entry_dict}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing unique entry for key {current_key}: {e}",
                        exc_info=True,
                    )
                    continue

        output_progress.close()
        output_duration = time.time() - output_start_time
        logger.info(
            f"Finished processing sorted output ({unique_lines_processed:,} unique positions) in {output_duration:.2f} seconds."
        )

        # --- Final partial shard ---
        if current_shard_batch:
            logger.info(
                f"Processing final batch for shard {shard_index} ({len(current_shard_batch)} records). Sampling..."
            )
            final_train_batch, final_test_samples = sample_and_split_batch(
                current_shard_batch, samples_per_shard_for_test
            )
            if final_test_samples:
                test_set_accumulated.extend(final_test_samples)
                total_positions_test_final += len(final_test_samples)
                logger.info(
                    f"Sampled {len(final_test_samples)} for test set from final batch."
                )

            if final_train_batch:
                shard_file_path = output_dir_path / f"shard_{shard_index}.parquet"
                written = write_batch_to_parquet(
                    final_train_batch, shard_file_path, ARROW_SCHEMA
                )
                logger.info(
                    f"Written final training shard {shard_index} ({written} records)"
                )
                total_positions_train_final += written

                # Update stats
                for entry in final_train_batch:
                    stat_elo = entry["elo"]
                    stat_depth = entry["depth"]
                    color = entry["fen"].split(" ")[1]
                    if color == "w":
                        white_stats.update(stat_elo)
                        white_depth_stats.update(stat_depth)
                    else:
                        black_stats.update(stat_elo)
                        black_depth_stats.update(stat_depth)
                    overall_stats.update(stat_elo)
                    overall_depth_stats.update(stat_depth)
            else:
                logger.warning(f"Final shard {shard_index} is empty after sampling.")
            current_shard_batch.clear()

        # Write test set
        if test_set_accumulated:
            logger.info(
                f"Writing accumulated test set ({len(test_set_accumulated)} records)..."
            )
            test_written = write_batch_to_parquet(
                test_set_accumulated, test_output_path, ARROW_SCHEMA
            )
            logger.info(
                f"Written test set with {test_written} records to {test_output_path}"
            )
            if test_written != total_positions_test_final:
                logger.warning(
                    f"Mismatch counted ({total_positions_test_final}) vs written ({test_written}) test samples."
                )
        else:
            logger.warning("No samples were collected for the test set.")

        # Check sort process status
        logger.info("Waiting for sort process to exit...")
        sort_proc.wait()
        stderr_data = ""
        if sort_proc.stderr and not sort_proc.stderr.closed:
            try:
                stderr_data = sort_proc.stderr.read()
            finally:
                try:
                    sort_proc.stderr.close()
                except Exception:
                    pass

        if sort_proc.returncode != 0:
            logger.error(f"Sort process failed with return code {sort_proc.returncode}")
            if stderr_data:
                logger.error(f"Sort process stderr:\n----\n{stderr_data}\n----")
            else:
                logger.error(
                    "Sort process stderr was empty or unreadable after failure."
                )
            if sort_proc.stdout and not sort_proc.stdout.closed:
                sort_proc.stdout.close()
            raise RuntimeError(
                f"External sort process failed (code {sort_proc.returncode})."
            )
        else:
            logger.info("External sort process completed successfully.")
            if stderr_data:
                logger.warning(
                    f"Sort process stderr (exit code 0):\n----\n{stderr_data}\n----"
                )

    except Exception as e:
        error_message = (
            f"An error occurred during the second pass: {e}\n{traceback.format_exc()}"
        )
        logger.error(error_message)
        if sort_proc and sort_proc.poll() is None:
            logger.warning("Attempting to terminate sort process due to error...")
            try:
                sort_proc.terminate()
                try:
                    sort_proc.wait(timeout=5)
                    logger.info("Sort process terminated gracefully.")
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Sort process did not terminate gracefully, killing."
                    )
                    sort_proc.kill()
                    sort_proc.wait()
                    logger.info("Sort process killed.")
            except Exception as term_err:
                logger.error(f"Error terminating/killing sort process: {term_err}")

    final_mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info("Second Pass complete.")
    logger.info(f"Unique positions found: {unique_lines_processed:,}")
    logger.info(f"Training positions written: {total_positions_train_final:,}")
    logger.info(f"Test positions written: {total_positions_test_final:,}")
    logger.info(
        f"Final Python Proc memory: {final_mem_mb:.2f} MB "
        f"(Delta: {final_mem_mb - initial_mem_mb:.2f} MB)"
    )

    # --- Prepare Metadata ---
    metadata = {
        "pgn_games_parsed": total_games if total_games > 0 else "Skipped or N/A",
        "input_positions_fed_to_sort": input_count_sort_actual,
        "unique_positions_found": unique_lines_processed,
        "train_set_size": total_positions_train_final,
        "test_set_size": total_positions_test_final,
        "samples_per_shard_for_test": samples_per_shard_for_test,
        "min_elo_threshold": min_elo_threshold,
        "positions_per_shard_target": positions_per_shard,
        # New: game-level outcomes
        "white_wins_count": white_win_count,
        "black_wins_count": black_win_count,
        "draws_count": draw_count,
        # ELO Stats
        "white_elo_stats_train": white_stats.to_dict(),
        "black_elo_stats_train": black_stats.to_dict(),
        "overall_elo_stats_train": overall_stats.to_dict(),
        # Depth Stats
        "white_depth_stats_train": white_depth_stats.to_dict(),
        "black_depth_stats_train": black_depth_stats.to_dict(),
        "overall_depth_stats_train": overall_depth_stats.to_dict(),
    }
    metadata_path = output_dir_path / "metadata.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata file: {e}")

    logger.info("--- Processing Finished ---")
    logger.info(f"Final sharded TRAINING Parquet data written to: {output_dir_path}")
    logger.info(f"Final TEST Parquet data written to: {test_output_path}")


if __name__ == "__main__":
    try:
        generate()
    except Exception as main_e:
        logger.critical(f"Critical error in main execution: {main_e}", exc_info=True)
        sys.exit(1)
