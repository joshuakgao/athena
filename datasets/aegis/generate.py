import json
import subprocess
import time  # For potential delays if needed
import traceback  # Import traceback module
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
from utils.logger import logger


# --- END MOCKING ---


# --- Configuration ---
raw_dir = "datasets/aegis/raw_data"
dir = "datasets/aegis/data"
temp_dir_path = Path("datasets/aegis/temp_parquet")  # Changed temp dir name
output_dir_path = Path(dir)
# Adjust batch sizes based on memory constraints when converting to DataFrames
# This replaces positions_per_temp_file for writing parquet
rows_per_temp_parquet_write = (
    1_000_000  # How many rows to batch before writing a temp parquet file
)
positions_per_shard = 10_000_000  # Rows per final parquet shard
min_elo_threshold = 2500
sort_memory = "4G"
sort_parallelism = 4
sort_temp_dir = None

# Define PyArrow schema for consistency (optional but recommended)
# Helps ensure data types are correct and consistent across files.
ARROW_SCHEMA = pa.schema(
    [
        ("fen", pa.string()),
        ("best_move", pa.string()),
        ("elo", pa.int32()),  # Use int32 for ELO
        ("history", pa.list_(pa.string())),
        ("bot", pa.string()),
    ]
)


class RunningStats:
    """Class to calculate running statistics without storing all values"""

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
        return 0
    try:
        df = pd.DataFrame(batch)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        pq.write_table(table, file_path)
        written_count = len(batch)
        batch.clear()  # Clear the batch after writing
        return written_count
    except Exception as e:
        logger.error(f"Error writing batch to {file_path}: {e}")
        logger.error(f"Problematic batch (first few items): {batch[:5]}")
        raise  # Re-raise the error to potentially stop the process


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
        temp_file_index = 709
        current_temp_batch = []  # Accumulate records here
        total_written_to_temp = 0

        pgn_files = sorted(list(Path(raw_dir).glob("*.pgn")))
        if not pgn_files:
            logger.warning(f"No PGN files found in {raw_dir}. Exiting.")
            return

        for pgn_path in tqdm(pgn_files, desc="Processing PGN files"):
            logger.info(f"Processing {pgn_path.name}...")
            games_in_file = 0
            positions_in_file_batch = (
                0  # Positions added to current batch from this file
            )

            try:
                with open(pgn_path, errors="replace") as pgn_file:
                    while True:
                        try:
                            game = chess.pgn.read_game(pgn_file)
                        except Exception as e:
                            logger.warning(
                                f"Error reading game in {pgn_path.name}: {e}. Skipping game."
                            )
                            continue

                        if game is None:
                            break

                        total_games += 1
                        games_in_file += 1

                        white_bot = game.headers.get("White", "Unknown")
                        black_bot = game.headers.get("Black", "Unknown")

                        try:
                            if white_bot == "Lc0":
                                white_elo = "3404"
                            else:
                                white_elo = game.headers.get("WhiteElo", "0")
                                white_elo = (
                                    int(white_elo) if white_elo.strip() else 0
                                )  # Handle empty strings

                            if black_bot == "Lc0":
                                black_elo = "3404"
                            else:
                                black_elo = game.headers.get("BlackElo", "0")
                                black_elo = (
                                    int(black_elo) if black_elo.strip() else 0
                                )  # Handle empty strings

                        except ValueError as e:
                            logger.warning(
                                f"Invalid ELO in headers for a game in {pgn_path.name}. Skipping game."
                            )
                            print(e)
                            continue

                        white_elo = int(white_elo)
                        black_elo = int(black_elo)
                        if (
                            white_elo < min_elo_threshold
                            and black_elo < min_elo_threshold
                        ):
                            continue

                        board = game.board()
                        last_seven_fens = deque(maxlen=7)

                        try:
                            for move in game.mainline_moves():
                                fen = board.fen()
                                board.push(move)

                                if not board.is_valid():
                                    logger.debug(f"Skipping invalid FEN: {fen}")
                                    continue

                                # format fen to remove en passant and move counter
                                fen_split = fen.split()
                                fen_split[3] = "-"
                                fen_split[4] = "0"
                                fen_split[5] = "1"
                                fen = " ".join(fen_split)

                                next_move_uci = move.uci()
                                active_color = fen.split()[1]
                                elo = white_elo if active_color == "w" else black_elo
                                bot = white_bot if active_color == "w" else black_bot

                                if elo >= min_elo_threshold:
                                    entry = {
                                        "fen": fen,
                                        "best_move": next_move_uci,
                                        "elo": elo,
                                        "history": list(last_seven_fens),
                                        "bot": bot,
                                    }
                                    current_temp_batch.append(entry)
                                    positions_in_file_batch += 1
                                    input_positions_estimate += 1

                                    # Write batch to Parquet if size threshold reached
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
                                        logger.info(
                                            f"Written temp batch to {temp_file_path.name} ({written} records)"
                                        )
                                        total_written_to_temp += written
                                        temp_file_index += 1

                                last_seven_fens.appendleft(
                                    fen
                                )  # History is FENs *before* the current one

                        except Exception as e:
                            logger.warning(
                                f"Error processing moves in game {total_games} from {pgn_path.name}: {e}"
                            )

            except FileNotFoundError:
                logger.error(f"PGN file not found: {pgn_path}. Skipping.")
                continue
            except Exception as e:
                logger.error(
                    f"Failed to process PGN file {pgn_path.name}: {e}", exc_info=False
                )
                continue

            logger.info(
                f"Finished {pgn_path.name}. Games processed: {games_in_file}. Positions added to batch: {positions_in_file_batch}."
            )

        # Write any remaining records in the last batch
        if current_temp_batch:
            temp_file_path = temp_dir_path / f"temp_{temp_file_index}.parquet"
            written = write_batch_to_parquet(
                current_temp_batch, temp_file_path, ARROW_SCHEMA
            )
            logger.info(
                f"Written final temp batch to {temp_file_path.name} ({written} records)"
            )
            total_written_to_temp += written

        logger.info(f"TOTAL PGN GAMES PARSED: {total_games}")
        logger.info(
            f"Total records written to temp Parquet files: {total_written_to_temp}"
        )
        if total_written_to_temp != input_positions_estimate:
            logger.warning(
                f"Mismatch between estimated positions ({input_positions_estimate}) and written ({total_written_to_temp})"
            )

        temp_files = sorted(list(temp_dir_path.glob("temp_*.parquet")))
        if not temp_files:
            logger.warning("No temporary Parquet files were generated. Exiting.")
            return
    else:
        logger.info(
            f"Found {len(temp_files)} existing temp Parquet files, skipping PGN processing (First Pass)."
        )
        logger.warning("Input position count will be estimated based on sort input.")

    # --- SECOND PASS: External Sort + Reduce ---
    logger.info("Starting Second Pass: Deduplication and Filtering using External Sort")
    process = psutil.Process()
    initial_mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Memory before sort pipe: {initial_mem_mb:.2f} MB")

    white_stats = RunningStats()
    black_stats = RunningStats()
    overall_stats = RunningStats()

    current_shard_batch = []  # Accumulate records for the current output shard
    shard_index = 0
    total_positions_final = 0
    input_count_sort = 0

    sort_command = [
        "sort",
        "-t",
        "\t",
        "-k1,1",
        "-k2,2",
        "-S",
        sort_memory,
        f"--parallel={sort_parallelism}",
    ]
    if sort_temp_dir:
        sort_command.extend(["-T", sort_temp_dir])

    logger.info(f"Using sort command: {' '.join(sort_command)}")

    sort_proc = None
    try:
        sort_proc = subprocess.Popen(
            sort_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=8192,
        )

        logger.info("Feeding data to external sort from temp Parquet files...")
        feeding_start_time = time.time()

        try:
            for temp_file_path in tqdm(temp_files, desc="Feeding temp files"):
                logger.debug(f"Processing temp file: {temp_file_path}")
                try:
                    # Read entire temp parquet file (adjust if memory is an issue)
                    # Consider pq.ParquetFile(temp_file_path).iter_batches(batch_size=...)
                    table = pq.read_table(temp_file_path)
                    # Convert to pandas DataFrame for easier row iteration
                    df = table.to_pandas()

                    for _, row in df.iterrows():
                        try:
                            entry_dict = row.to_dict()

                            if "history" in entry_dict:
                                history_value = entry_dict["history"]
                                if isinstance(history_value, np.ndarray):
                                    # Convert numpy array to Python list
                                    entry_dict["history"] = history_value.tolist()
                                elif not isinstance(history_value, list):
                                    # If it's not an ndarray or list, maybe log warning or force to list
                                    logger.warning(
                                        f"History field in {temp_file_path} is not a list or ndarray: {type(history_value)}. Attempting conversion or defaulting to empty list."
                                    )
                                    try:
                                        entry_dict["history"] = list(history_value)
                                    except TypeError:
                                        entry_dict["history"] = []
                            else:
                                entry_dict["history"] = (
                                    []
                                )  # Ensure history key exists as a list
                            fen = entry_dict["fen"]
                            history_list = entry_dict["history"]
                            history_str = "|".join(history_list)
                            elo = entry_dict["elo"]

                            # Serialize the original dictionary to JSON to pass through sort
                            # Ensure ensure_ascii=False if you have non-ASCII in 'bot' names
                            original_json_str = json.dumps(
                                entry_dict, ensure_ascii=False
                            )

                            # Output format: FEN \t HistoryStr \t ELO \t OriginalJSON \n
                            tsv_line = (
                                f"{fen}\t{history_str}\t{elo}\t{original_json_str}\n"
                            )

                            sort_proc.stdin.write(tsv_line)
                            input_count_sort += 1

                        except (KeyError, TypeError) as e:
                            logger.warning(
                                f"Skipping malformed row in {temp_file_path}: {row} - Error: {e}"
                            )
                            continue
                        except BrokenPipeError:
                            logger.error(
                                "Broken Pipe Error: Sort process terminated unexpectedly while feeding data."
                            )
                            stderr_data_on_break = ""
                            if sort_proc.stderr:
                                try:
                                    stderr_data_on_break = sort_proc.stderr.read()
                                except Exception as read_err:
                                    logger.warning(
                                        f"Could not read stderr after broken pipe: {read_err}"
                                    )
                            if stderr_data_on_break:
                                logger.error(
                                    f"Sort stderr output captured immediately after break:\n{stderr_data_on_break}"
                                )
                            raise
                        except Exception as e:
                            logger.error(
                                f"Unexpected error writing to sort stdin for row {row}: {e}",
                                exc_info=False,
                            )
                            raise  # Re-raise to stop processing

                    # Memory check (optional)
                    if input_count_sort % 10 == 0:
                        mem_mb = process.memory_info().rss / (1024 * 1024)
                        logger.debug(
                            f"Fed {input_count_sort:,} lines. Python Proc Mem: {mem_mb:.2f} MB"
                        )

                except Exception as read_e:
                    logger.error(
                        f"Failed to read or process {temp_file_path}: {read_e}"
                    )
                    # Decide whether to skip the file or stop
                    continue  # Skip this temp file

        finally:
            if sort_proc and sort_proc.stdin:
                try:
                    sort_proc.stdin.close()
                except OSError as e:
                    logger.warning(
                        f"Could not close sort stdin (might be already closed): {e}"
                    )

        feeding_duration = time.time() - feeding_start_time
        logger.info(
            f"Finished feeding {input_count_sort:,} lines to sort in {feeding_duration:.2f} seconds. Waiting for sort to complete and processing output..."
        )
        mem_after_feed_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(
            f"Memory after feeding, before reduction: {mem_after_feed_mb:.2f} MB"
        )

        # --- Process the sorted output ---
        current_key = None
        current_max_elo = -1
        current_best_line_json = (
            None  # This now holds the JSON string of the best record
        )

        output_start_time = time.time()
        output_progress = tqdm(
            sort_proc.stdout,
            desc=f"Reducing sorted data (Shard {shard_index})",
            unit=" lines",
            mininterval=2.0,
        )

        for sorted_line in output_progress:
            try:
                parts = sorted_line.strip().split("\t", 3)
                if len(parts) != 4:
                    logger.warning(
                        f"Skipping malformed sorted line (expected 4 fields): {sorted_line.strip()}"
                    )
                    continue
                fen, history_str, elo_str, original_json_str = parts
                elo = int(elo_str)  # Use ELO from TSV for comparison
                key = (fen, history_str)
            except ValueError as e:
                logger.warning(
                    f"Skipping malformed sorted line (ValueError): {sorted_line.strip()} - Error: {e}"
                )
                continue
            except Exception as e:
                logger.warning(
                    f"Skipping malformed sorted line (Other Error): {sorted_line.strip()} - Error: {e}"
                )
                continue

            if key == current_key:
                if elo > current_max_elo:
                    current_max_elo = elo
                    current_best_line_json = original_json_str  # Keep the JSON string
            else:
                # New key: Process the best entry for the previous key
                if current_key is not None and current_best_line_json is not None:
                    try:
                        # Parse the JSON to get the full record dictionary
                        best_entry_dict = json.loads(current_best_line_json)

                        # Add the best dictionary to the current shard batch
                        current_shard_batch.append(best_entry_dict)
                        total_positions_final += 1

                        # Update stats using the max ELO found (which is in best_entry_dict)
                        stat_elo = best_entry_dict[
                            "elo"
                        ]  # Use ELO from the record itself
                        color = best_entry_dict["fen"].split(" ")[1]
                        if color == "w":
                            white_stats.update(stat_elo)
                        else:
                            black_stats.update(stat_elo)
                        overall_stats.update(stat_elo)

                        # Handle sharding: Write Parquet file when batch is full
                        if len(current_shard_batch) >= positions_per_shard:
                            shard_file_path = (
                                output_dir_path / f"shard_{shard_index}.parquet"
                            )
                            written = write_batch_to_parquet(
                                current_shard_batch, shard_file_path, ARROW_SCHEMA
                            )
                            logger.info(
                                f"Written output shard {shard_index} to {shard_file_path.name} ({written} records)"
                            )
                            shard_index += 1
                            output_progress.set_description(
                                f"Reducing sorted data (Shard {shard_index})"
                            )

                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.warning(
                            f"Could not process best entry for key {current_key}: {e} - JSON: {current_best_line_json[:100]}..."
                        )
                    except (
                        Exception
                    ) as e:  # Catch potential write_batch_to_parquet errors
                        logger.error(f"Error writing shard {shard_index}: {e}")
                        raise  # Stop processing if shard writing fails

                # Reset for the new key
                current_key = key
                current_max_elo = elo
                current_best_line_json = original_json_str

        # --- After loop: Process the very last best entry ---
        if current_key is not None and current_best_line_json is not None:
            try:
                best_entry_dict = json.loads(current_best_line_json)
                current_shard_batch.append(best_entry_dict)
                total_positions_final += 1

                stat_elo = best_entry_dict["elo"]
                color = best_entry_dict["fen"].split(" ")[1]
                if color == "w":
                    white_stats.update(stat_elo)
                else:
                    black_stats.update(stat_elo)
                overall_stats.update(stat_elo)

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.warning(
                    f"Could not process final best entry for stats ({current_key}): {e} - JSON: {current_best_line_json[:100]}..."
                )

        output_progress.close()
        output_duration = time.time() - output_start_time
        logger.info(
            f"Finished processing sorted output in {output_duration:.2f} seconds."
        )

        # Write the final remaining batch to the last shard file
        if current_shard_batch:
            shard_file_path = output_dir_path / f"shard_{shard_index}.parquet"
            written = write_batch_to_parquet(
                current_shard_batch, shard_file_path, ARROW_SCHEMA
            )
            logger.info(
                f"Written final output shard {shard_index} to {shard_file_path.name} ({written} records)"
            )

        # --- Check Sort Process Exit Status ---
        sort_proc.wait()
        stderr_data = ""
        if sort_proc.stderr:
            try:
                stderr_data = sort_proc.stderr.read()
            except Exception as e:
                logger.warning(f"Could not read sort stderr after wait: {e}")
            finally:
                try:
                    sort_proc.stderr.close()
                except Exception as e:
                    logger.warning(f"Error closing sort stderr pipe: {e}")

        if sort_proc.returncode != 0:
            logger.error(
                f"External sort process failed with return code {sort_proc.returncode}"
            )
            if stderr_data:
                logger.error(f"Sort process stderr:\n----\n{stderr_data}\n----")
            else:
                logger.error("Sort process stderr was empty or could not be read.")
            raise RuntimeError(
                f"External sort process failed (code {sort_proc.returncode}). Check logs for details."
            )
        else:
            logger.info("External sort process completed successfully.")
            if stderr_data:
                logger.warning(
                    f"Sort process stderr (returned 0):\n----\n{stderr_data}\n----"
                )

    except Exception as e:
        error_message = (
            f"An error occurred during the second pass: {e}\n{traceback.format_exc()}"
        )
        logger.error(error_message)
        # No explicit output file handle to close here, batches are handled internally

        if sort_proc and sort_proc.poll() is None:
            logger.warning("Attempting to terminate sort process due to error...")
            stderr_data_on_except = ""
            # ...(rest of the sort process termination logic - unchanged)
            if sort_proc.stderr and not sort_proc.stderr.closed:
                try:
                    stderr_data_on_except = sort_proc.stderr.read()
                except Exception as stderr_e:
                    logger.error(
                        f"Could not read sort stderr during exception handling: {stderr_e}"
                    )
            if stderr_data_on_except:
                logger.error(
                    f"Sort process stderr captured during exception handling:\n----\n{stderr_data_on_except}\n----"
                )
            try:
                sort_proc.terminate()
                try:
                    sort_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Sort process did not terminate gracefully, sending SIGKILL."
                    )
                    sort_proc.kill()
                    sort_proc.wait()
                logger.warning("Sort process terminated.")
            except Exception as term_err:
                logger.error(f"Error trying to terminate sort process: {term_err}")
            finally:
                if sort_proc.stdin and not sort_proc.stdin.closed:
                    sort_proc.stdin.close()
                if sort_proc.stdout and not sort_proc.stdout.closed:
                    sort_proc.stdout.close()
                if sort_proc.stderr and not sort_proc.stderr.closed:
                    sort_proc.stderr.close()
        raise e

    # --- Final Steps ---
    final_mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(
        f"Second Pass complete. Total unique positions written: {total_positions_final:,}"
    )
    logger.info(
        f"Final Python Proc memory usage: {final_mem_mb:.2f} MB (Delta: {final_mem_mb - initial_mem_mb:.2f} MB)"
    )

    # Prepare metadata (remains JSON)
    metadata = {
        "pgn_games_parsed": total_games if total_games > 0 else "Skipped or N/A",
        "total_positions": total_positions_final,
        "min_elo": min_elo_threshold,
        "positions_per_shard": positions_per_shard,  # Note: This is the target, actual might vary slightly
        "white_elo_stats": white_stats.to_dict(),
        "black_elo_stats": black_stats.to_dict(),
        "overall_elo_stats": overall_stats.to_dict(),
    }

    metadata_path = output_dir_path / "metadata.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata file: {e}")

    logger.info("--- Processing Finished ---")
    logger.info(f"Final sharded Parquet data written to: {output_dir_path}")

    # Optional: Clean up temp parquet files
    # logger.info("Cleaning up temporary Parquet files...")
    # cleanup_count = 0
    # try:
    #     for temp_file in temp_dir_path.glob("temp_*.parquet"):
    #         temp_file.unlink()
    #         cleanup_count += 1
    #     logger.info(f"Removed {cleanup_count} temporary Parquet files.")
    #     # try:
    #     #     temp_dir_path.rmdir()
    #     #     logger.info("Removed temporary directory.")
    #     # except OSError as e:
    #     #     logger.warning(f"Could not remove temporary directory: {e}")
    # except Exception as e:
    #      logger.error(f"Error during temporary file cleanup: {e}")


def validate(fen, history):
    # Convert history to tuple of FEN strings for consistent comparison
    count = 0

    aegis = sorted(output_dir_path.glob("*.parquet"))
    for parquet_file in tqdm(aegis):
        print(f"Current count: {count}")  # More informative progress
        try:
            df = pq.read_table(parquet_file).to_pandas()

            # First filter by the current FEN (fast string comparison)
            fen_matches = df[df["fen"] == fen]

            # Then compare histories
            for _, row in fen_matches.iterrows():
                print(row["fen"])
                # Ensure we're comparing tuples of FEN strings
                stored_history = row["history"]
                print(stored_history)
                print(history)
                if list(stored_history) == history:
                    count += 1
                    if count > 1:
                        logger.warning(
                            f"Duplicate entry found in {parquet_file}:\n"
                            f"FEN: {fen}\n"
                            f"History: {history}"
                        )
                        return False  # Early exit if duplicate found

        except Exception as e:
            logger.error(f"Error processing {parquet_file}: {e}")
            continue  # Skip to next file if error occurs

    # Return validation result
    if count == 1:
        return True
    elif count == 0:
        logger.warning("No matching entry found in any parquet file")
        return False
    else:
        logger.warning(f"Found {count} duplicate entries")
        return False


if __name__ == "__main__":
    generate()
