import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pandas as pd
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
        # History is now a list of the last 7 FEN strings
        ("history", pa.list_(pa.string())),
        ("bot", pa.string()),
    ]
)


class RunningStats:
    """Class to calculate running statistics without storing all values"""

    # (Implementation remains the same as before)
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
    # (Implementation remains the same as before)
    if not batch:
        logger.warning(f"Attempted to write an empty batch to {file_path}. Skipping.")
        return 0
    try:
        # Ensure history is list, not numpy array (can happen from pandas conversion)
        for record in batch:
            if "history" in record and isinstance(record["history"], np.ndarray):
                record["history"] = record["history"].tolist()

        table = pa.Table.from_pylist(
            batch, schema=schema
        )  # Use from_pylist for direct dict list
        pq.write_table(table, file_path)
        written_count = len(batch)
        # batch.clear() # Don't clear here, caller manages the batch
        return written_count
    except pa.ArrowInvalid as e:
        logger.error(f"ArrowInvalid error writing batch to {file_path}: {e}")
        # Log details about the problematic batch for debugging
        logger.error(f"Schema: {schema}")
        if batch:
            logger.error(f"First item type: {type(batch[0])}")
            logger.error(f"First item keys: {batch[0].keys()}")
            # Check types of values in the first item
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
    # (Implementation remains the same as before)
    if not batch:
        return [], []

    actual_samples = min(num_samples, len(batch))
    if actual_samples == 0:
        return batch, []

    sample_indices = random.sample(range(len(batch)), actual_samples)
    test_samples = [batch[i] for i in sample_indices]

    # Create the training batch by excluding sampled indices
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
        # Return original or raise error if FEN is invalid
        logger.warning(
            f"Encountered potentially invalid FEN for normalization: {fen_string}"
        )
        return fen_string  # Or handle error differently


def generate():
    output_dir_path.mkdir(parents=True, exist_ok=True)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    # --- FIRST PASS: Write all FENs to temp Parquet files ---
    temp_files = sorted(list(temp_dir_path.glob("temp_*.parquet")))

    total_games = 0
    input_positions_estimate = 0

    # Only process PGN files if no temp files exist
    if not temp_files:
        logger.info(
            "No temporary Parquet files found. Starting First Pass: Processing PGNs..."
        )
        temp_file_index = 0
        current_temp_batch = []
        total_written_to_temp = 0

        pgn_files = sorted(list(Path(raw_dir).glob("*.pgn")))
        if not pgn_files:
            logger.warning(f"No PGN files found in {raw_dir}. Exiting.")
            return

        for pgn_path in tqdm(pgn_files, desc="Processing PGN files"):
            logger.info(f"Processing {pgn_path.name}...")
            games_in_file = 0
            positions_in_file_batch = 0

            try:
                with open(pgn_path, errors="replace", encoding="utf-8") as pgn_file:
                    game_offset = pgn_file.tell()
                    while True:
                        # (Error handling for reading games remains the same)
                        try:
                            start_offset = pgn_file.tell()
                            game = chess.pgn.read_game(pgn_file)
                            if game is None and pgn_file.tell() == start_offset:
                                line = pgn_file.readline()
                                if not line:
                                    break
                                logger.warning(
                                    f"Empty read at offset {start_offset} in {pgn_path.name}, skipping line."
                                )
                                continue
                        except (ValueError, RuntimeError) as e:
                            logger.warning(
                                f"Error reading game headers/structure near offset {game_offset} in {pgn_path.name}: {e}. Attempting recovery."
                            )
                            # ... (recovery logic remains the same) ...
                            pgn_file.seek(game_offset)
                            try:
                                while True:
                                    line = pgn_file.readline()
                                    if not line:
                                        game = None
                                        break
                                    if line.startswith("[Event "):
                                        pgn_file.seek(game_offset)
                                        game = chess.pgn.read_game(pgn_file)
                                        break
                                    game_offset = pgn_file.tell()
                            except Exception as seek_e:
                                logger.error(
                                    f"Recovery attempt failed in {pgn_path.name}: {seek_e}. Skipping rest of file."
                                )
                                game = None
                        except Exception as e:
                            logger.error(
                                f"Unexpected error reading game near offset {game_offset} in {pgn_path.name}: {e}. Skipping game.",
                                exc_info=False,
                            )
                            # ... (recovery logic remains the same) ...
                            continue

                        if game is None:
                            break

                        game_offset = pgn_file.tell()
                        total_games += 1
                        games_in_file += 1

                        # (ELO Handling remains the same)
                        white_bot = game.headers.get("White", "Unknown")
                        black_bot = game.headers.get("Black", "Unknown")
                        white_elo_str = game.headers.get("WhiteElo", "0")
                        black_elo_str = game.headers.get("BlackElo", "0")
                        # ... (logic to parse or assign fixed ELOs) ...
                        if white_bot == "Lc0":
                            white_elo = 3404
                        elif white_bot == "Stockfish 101217 64 BMI2":
                            white_elo = 3529
                        else:
                            try:
                                white_elo = (
                                    int(white_elo_str) if white_elo_str.strip() else 0
                                )
                            except ValueError:
                                logger.warning(
                                    f"Invalid WhiteElo '{white_elo_str}'... Using 0."
                                )
                                white_elo = 0

                        if black_bot == "Lc0":
                            black_elo = 3404
                        elif black_bot == "Stockfish 101217 64 BMI2":
                            black_elo = 3529
                        else:
                            try:
                                black_elo = (
                                    int(black_elo_str) if black_elo_str.strip() else 0
                                )
                            except ValueError:
                                logger.warning(
                                    f"Invalid BlackElo '{black_elo_str}'... Using 0."
                                )
                                black_elo = 0

                        if (
                            white_elo < min_elo_threshold
                            and black_elo < min_elo_threshold
                        ):
                            continue

                        board = game.board()
                        # --- HISTORY CHANGE: Use deque to store last 7 FENs ---
                        last_seven_fens = deque(maxlen=7)
                        # --- END HISTORY CHANGE ---

                        move_index = 0
                        try:
                            for move in game.mainline_moves():
                                move_index += 1
                                # --- Get state BEFORE the move ---
                                current_fen_unnormalized = board.fen()
                                next_move_uci = move.uci()

                                # --- Normalize the FEN of the state we are recording ---
                                norm_fen = normalize_fen(current_fen_unnormalized)
                                if (
                                    norm_fen == current_fen_unnormalized
                                    and len(current_fen_unnormalized.split()) != 6
                                ):
                                    logger.warning(
                                        f"Skipping position due to invalid pre-move FEN: {current_fen_unnormalized} in game {total_games}"
                                    )
                                    # Need to push move to advance board state even if we skip recording
                                    try:
                                        board.push(move)
                                    except ValueError:
                                        break  # Stop if move becomes illegal
                                    continue  # Skip recording this position

                                # Check board validity *before* pushing the move
                                if not board.is_valid():
                                    logger.warning(
                                        f"Board invalid *before* move {next_move_uci} in game {total_games}. FEN: {current_fen_unnormalized}. Skipping rest of game."
                                    )
                                    break

                                # --- Determine ELO/Bot based on whose turn it is in current_fen ---
                                fen_parts = (
                                    current_fen_unnormalized.split()
                                )  # Use unnormalized to check turn reliably
                                active_color = (
                                    fen_parts[1] if len(fen_parts) > 1 else "w"
                                )  # Default to white if FEN is weird
                                elo = white_elo if active_color == "w" else black_elo
                                bot = white_bot if active_color == "w" else black_bot

                                # --- Record the position if ELO is sufficient ---
                                if elo >= min_elo_threshold:
                                    entry = {
                                        "fen": norm_fen,  # The normalized FEN *before* the move
                                        "best_move": next_move_uci,  # The move played from this state
                                        "elo": elo,
                                        # --- HISTORY CHANGE: Store the list of previous FENs ---
                                        "history": list(last_seven_fens),
                                        # --- END HISTORY CHANGE ---
                                        "bot": bot,
                                    }
                                    current_temp_batch.append(entry)
                                    positions_in_file_batch += 1
                                    input_positions_estimate += 1

                                    # (Batch writing logic remains the same)
                                    if (
                                        len(current_temp_batch)
                                        >= rows_per_temp_parquet_write
                                    ):
                                        temp_file_path = (
                                            temp_dir_path
                                            / f"temp_{temp_file_index}.parquet"
                                        )
                                        written = write_batch_to_parquet(
                                            current_temp_batch,
                                            temp_file_path,
                                            ARROW_SCHEMA,
                                        )
                                        if written > 0:
                                            logger.info(
                                                f"Written temp batch to {temp_file_path.name} ({written} records)"
                                            )
                                            total_written_to_temp += written
                                            temp_file_index += 1
                                        current_temp_batch.clear()

                                # --- HISTORY CHANGE: Add the *processed* normalized FEN to history deque ---
                                # Do this *after* potentially creating the entry for this FEN
                                last_seven_fens.appendleft(norm_fen)
                                # --- END HISTORY CHANGE ---

                                # --- Push the move to advance the board state for the NEXT iteration ---
                                try:
                                    board.push(move)
                                except ValueError as e:
                                    logger.warning(
                                        f"Illegal move '{next_move_uci}' (move #{move_index}) encountered in game {total_games}. FEN: {norm_fen}. Error: {e}. Skipping rest of game."
                                    )
                                    break

                        except (AttributeError, ValueError, IndexError) as e:
                            logger.warning(
                                f"Error processing moves in game {total_games} from {pgn_path.name} near move #{move_index}: {e}. Last valid FEN: {board.fen() if 'board' in locals() else 'N/A'}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Unexpected error during move processing loop for game {total_games}: {e}",
                                exc_info=True,
                            )

            # (Outer file processing error handling remains the same)
            except FileNotFoundError:
                logger.error(f"PGN file not found: {pgn_path}. Skipping.")
                continue
            except Exception as e:
                logger.error(
                    f"Failed to process PGN file {pgn_path.name}: {e}", exc_info=True
                )
                continue

            logger.info(
                f"Finished {pgn_path.name}. Games: {games_in_file}. Positions added: {positions_in_file_batch}."
            )

        # (Final temp batch writing remains the same)
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
                f"Mismatch: estimated positions ({input_positions_estimate:,}), written ({total_written_to_temp:,})."
            )

        temp_files = sorted(list(temp_dir_path.glob("temp_*.parquet")))
        if not temp_files:
            logger.warning("No temporary Parquet files were generated. Exiting.")
            return
    else:
        # (Skipping PGN processing if temp files exist remains the same)
        logger.info(
            f"Found {len(temp_files)} existing temp Parquet files, skipping PGN processing (First Pass)."
        )
        logger.info("Estimating input position count from existing temp files...")
        input_count_sort_est = 0
        for f in tqdm(temp_files, desc="Estimating size"):
            try:
                input_count_sort_est += pq.read_metadata(f).num_rows
            except Exception as e:
                logger.warning(f"Could not read metadata for {f.name}: {e}.")
        logger.info(f"Estimated input positions for sort: {input_count_sort_est:,}")
        input_positions_estimate = input_count_sort_est

    # --- SECOND PASS: External Sort + Reduce ---
    # This part remains identical to the previous version where test sets are
    # sampled per shard just before writing. The change to FEN history
    # doesn't affect the logic here, only the *content* of the history string
    # used for sorting and stored in the final Parquet files.
    logger.info("Starting Second Pass: Deduplication and Filtering using External Sort")
    process = psutil.Process()
    initial_mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Memory before sort pipe: {initial_mem_mb:.2f} MB")

    white_stats = RunningStats()
    black_stats = RunningStats()
    overall_stats = RunningStats()  # Stats based on final *training* data

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
        "-k1,1",
        "-k3,3nr",
        "-k2,2",  # Keep FEN, ELO desc, History
        "-S",
        sort_memory,
        f"--parallel={sort_parallelism}",
        "--stable",
    ]
    if sort_temp_dir:
        sort_command.extend(["-T", sort_temp_dir])

    logger.info(f"Using sort command: {' '.join(sort_command)}")
    logger.info(
        f"Sorting primarily by FEN, secondarily by ELO (desc), tertiarily by history (FEN list)."
    )

    sort_proc = None
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

        # --- Feed data to sort ---
        try:
            for temp_file_path in tqdm(temp_files, desc="Feeding temp files"):
                logger.debug(f"Processing temp file: {temp_file_path}")
                try:
                    table = pq.read_table(temp_file_path)
                    df = table.to_pandas()

                    for _, row in df.iterrows():
                        try:
                            entry_dict = row.to_dict()

                            # --- Data Validation and Formatting ---
                            if not all(
                                k in entry_dict
                                for k in ["fen", "best_move", "elo", "history", "bot"]
                            ):
                                logger.warning(
                                    f"Skipping row with missing keys in {temp_file_path}: {entry_dict}"
                                )
                                continue

                            history_value = entry_dict["history"]
                            if isinstance(history_value, np.ndarray):
                                entry_dict["history"] = history_value.tolist()
                            elif not isinstance(history_value, list):
                                entry_dict["history"] = (
                                    []
                                )  # Default to empty if not list

                            # Ensure all history elements are strings (should be FENs now)
                            entry_dict["history"] = [
                                str(h) for h in entry_dict["history"]
                            ]

                            fen = str(entry_dict["fen"])
                            history_list = entry_dict["history"]
                            # --- HISTORY CHANGE: Use a separator unlikely in FENs ---
                            history_str = "||".join(
                                history_list
                            )  # Join FENs for sort key
                            # --- END HISTORY CHANGE ---
                            elo = int(entry_dict["elo"])
                            best_move = str(entry_dict["best_move"])
                            bot = str(entry_dict["bot"])

                            clean_entry = {
                                "fen": fen,
                                "best_move": best_move,
                                "elo": elo,
                                "history": history_list,
                                "bot": bot,
                            }
                            original_json_str = json.dumps(
                                clean_entry, ensure_ascii=False, separators=(",", ":")
                            )
                            tsv_line = (
                                f"{fen}\t{history_str}\t{elo}\t{original_json_str}\n"
                            )

                            sort_proc.stdin.write(tsv_line)
                            input_count_sort_actual += 1

                        # (Error handling for row processing remains the same)
                        except (KeyError, TypeError, ValueError) as e:
                            logger.warning(
                                f"Skipping malformed row during prep for sort in {temp_file_path}: {row} - Error: {e}"
                            )
                            continue
                        except BrokenPipeError:
                            # ... (Broken pipe handling) ...
                            logger.error(
                                "Broken Pipe Error: Sort process terminated unexpectedly while feeding data."
                            )
                            # ... (stderr reading attempt) ...
                            raise
                        except Exception as e:
                            # ... (Other exception handling) ...
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
            # (Closing sort stdin remains the same)
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
        current_key = None  # Key is still (FEN, history_str)
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
        unique_lines_processed = 0

        for sorted_line in output_progress:
            try:
                parts = sorted_line.strip().split("\t", 3)
                if len(parts) != 4:
                    logger.warning(
                        f"Skipping malformed sorted line (expected 4 fields): {sorted_line.strip()[:100]}..."
                    )
                    continue
                fen, history_str, elo_str, original_json_str = parts
                key = (fen, history_str)  # Deduplication key

            except Exception as e:
                logger.warning(
                    f"Skipping malformed sorted line (Error parsing fields): {sorted_line.strip()[:100]}... - Error: {e}"
                )
                continue

            # --- Deduplication Logic (remains the same) ---
            if key != current_key:
                unique_lines_processed += 1
                current_key = key
                try:
                    best_entry_dict = json.loads(original_json_str)
                    current_shard_batch.append(
                        best_entry_dict
                    )  # Add best entry to batch

                    # --- Shard Writing and Test Set Sampling (remains the same) ---
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
                            # --- Update Stats (remains the same) ---
                            for entry in train_batch_part:
                                stat_elo = entry["elo"]
                                color = entry["fen"].split(" ")[1]
                                if color == "w":
                                    white_stats.update(stat_elo)
                                else:
                                    black_stats.update(stat_elo)
                                overall_stats.update(stat_elo)
                        else:
                            logger.warning(
                                f"Shard {shard_index} is empty after sampling."
                            )

                        current_shard_batch.clear()
                        shard_index += 1
                        output_progress.set_description(
                            f"Reducing sorted data (Shard {shard_index})"
                        )

                # (Error handling for JSON parsing, KeyErrors, etc. remains the same)
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

            # Implicitly ignore line if key == current_key (duplicate)

        output_progress.close()
        output_duration = time.time() - output_start_time
        logger.info(
            f"Finished processing sorted output ({unique_lines_processed:,} unique positions found) in {output_duration:.2f} seconds."
        )

        # --- Process the final remaining batch (remains the same) ---
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
                # Update stats for final batch
                for entry in final_train_batch:
                    stat_elo = entry["elo"]
                    color = entry["fen"].split(" ")[1]
                    if color == "w":
                        white_stats.update(stat_elo)
                    else:
                        black_stats.update(stat_elo)
                    overall_stats.update(stat_elo)
            else:
                logger.warning(f"Final shard {shard_index} is empty after sampling.")
            current_shard_batch.clear()

        # --- Write the accumulated test set (remains the same) ---
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

        # --- Check Sort Process Exit Status (remains the same) ---
        logger.info("Waiting for sort process to exit...")
        sort_proc.wait()
        # ... (stderr reading and return code checking) ...
        stderr_data = ""
        if sort_proc.stderr and not sort_proc.stderr.closed:
            try:
                stderr_data = sort_proc.stderr.read()
            finally:
                try:
                    sort_proc.stderr.close()
                except Exception:
                    pass  # Ignore errors closing already potentially closed pipe

        if sort_proc.returncode != 0:
            logger.error(
                f"External sort process failed with return code {sort_proc.returncode}"
            )
            if stderr_data:
                logger.error(f"Sort process stderr:\n----\n{stderr_data}\n----")
            else:
                logger.error(
                    "Sort process stderr was empty or unreadable after failure."
                )
            if sort_proc.stdout and not sort_proc.stdout.closed:
                sort_proc.stdout.close()
            if sort_proc.stdin and not sort_proc.stdin.closed:
                sort_proc.stdin.close()
            raise RuntimeError(
                f"External sort process failed (code {sort_proc.returncode})."
            )
        else:
            logger.info("External sort process completed successfully.")
            if stderr_data:
                logger.warning(
                    f"Sort process stderr reported (exit code 0):\n----\n{stderr_data}\n----"
                )

    # (Overall exception handling for second pass remains the same)
    except Exception as e:
        error_message = (
            f"An error occurred during the second pass: {e}\n{traceback.format_exc()}"
        )
        logger.error(error_message)
        # ... (Attempt to terminate sort process) ...
        if sort_proc and sort_proc.poll() is None:
            logger.warning("Attempting to terminate sort process due to error...")
            # ... (stderr reading and terminate/kill logic) ...
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

    # --- Final Steps ---
    # (Logging final counts and memory usage remains the same)
    final_mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info("Second Pass complete.")
    logger.info(f"Unique positions found: {unique_lines_processed:,}")
    logger.info(f"Training positions written: {total_positions_train_final:,}")
    logger.info(f"Test positions written: {total_positions_test_final:,}")
    logger.info(
        f"Final Python Proc memory: {final_mem_mb:.2f} MB (Delta: {final_mem_mb - initial_mem_mb:.2f} MB)"
    )

    # --- Prepare Metadata (remains the same structure) ---
    metadata = {
        "pgn_games_parsed": total_games if total_games > 0 else "Skipped or N/A",
        "input_positions_fed_to_sort": input_count_sort_actual,
        "unique_positions_found": unique_lines_processed,
        "train_set_size": total_positions_train_final,
        "test_set_size": total_positions_test_final,
        "samples_per_shard_for_test": samples_per_shard_for_test,
        "min_elo_threshold": min_elo_threshold,
        "positions_per_shard_target": positions_per_shard,
        "white_elo_stats_train": white_stats.to_dict(),
        "black_elo_stats_train": black_stats.to_dict(),
        "overall_elo_stats_train": overall_stats.to_dict(),
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
