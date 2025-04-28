import json
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.logger import logger

# --- Configuration ---
raw_dir = "datasets/aegis/raw_data"
dir = "datasets/aegis/data"
output_dir_path = Path(dir)
test_output_path = output_dir_path / "test.parquet"
positions_per_shard = 1_000_000
min_depth_threshold = 25
max_depth_threshold = 40
samples_per_shard_for_test = 100

# Define PyArrow schema
ARROW_SCHEMA = pa.schema(
    [
        ("fen", pa.string()),
        ("best_move", pa.string()),
        ("depth", pa.int32()),
        ("eval", pa.int32()),
    ]
)


def write_batch_to_parquet(batch, file_path, schema):
    """Helper function to write a batch of records to a Parquet file."""
    if not batch:
        logger.warning(f"Attempted to write an empty batch to {file_path}. Skipping.")
        return 0
    try:
        table = pa.Table.from_pylist(batch, schema=schema)
        pq.write_table(table, file_path)
        return len(batch)
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


def parse_lichess_eval_jsonl(jsonl_path):
    """Parse Lichess evaluation JSONL file and yield processed records."""
    file_size = jsonl_path.stat().st_size
    with open(jsonl_path, "rb") as f:
        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc="Reading JSONL"
        ) as pbar:
            for raw_bytes in f:
                line = raw_bytes.decode("utf-8").strip()
                pbar.update(len(raw_bytes))

                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON: {e}")
                    continue

                fen = data.get("fen")
                evals_list = data.get("evals", [])
                if not fen or not evals_list:
                    continue

                # Ensure FEN includes turn counters (set to 0 and 1 if missing)
                fen_parts = fen.split()
                if len(fen_parts) < 6:
                    fen = " ".join(fen_parts + ["0", "1"])

                # Get active color from FEN (part after the board, index 1)
                active_color = fen_parts[1] if len(fen_parts) > 1 else "w"

                # Filter evaluations to include only those within the depth range
                valid_evals = [
                    e
                    for e in evals_list
                    if min_depth_threshold <= e.get("depth", 0) <= max_depth_threshold
                ]

                valid_evals = sorted(
                    valid_evals, key=lambda e: e.get("depth", 0), reverse=True
                )

                if not valid_evals:
                    continue

                best_eval = valid_evals[0]

                if not best_eval:
                    continue

                depth = best_eval["depth"]
                pv = best_eval["pvs"][0]

                # Handle depth and evaluation
                if "cp" in pv:
                    eval = pv["cp"]
                elif "mate" in pv:
                    eval = 10000 if active_color == "w" else -10000
                else:
                    continue

                # Extract the first move from the "line" key as the best move
                best_move = pv["line"].split()[0]

                yield {
                    "fen": fen,
                    "best_move": best_move,
                    "depth": depth,
                    "eval": eval,
                }


def generate():
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Stats tracking
    stats = {
        "total_positions": 0,
        "train_positions": 0,
        "test_positions": 0,
        "depth_stats": {"min": float("inf"), "max": float("-inf"), "sum": 0},
    }

    current_shard_batch = []
    shard_index = 0
    test_set_accumulated = []

    jsonl_files = list(Path(raw_dir).glob("*.jsonl"))
    if not jsonl_files:
        logger.error("No JSONL files found in raw_data directory")
        return

    for jfile in jsonl_files:
        logger.info(f"Processing JSONL file: {jfile.name}")

        for record in parse_lichess_eval_jsonl(jfile):
            stats["total_positions"] += 1
            stats["depth_stats"]["min"] = min(
                stats["depth_stats"]["min"], record["depth"]
            )
            stats["depth_stats"]["max"] = max(
                stats["depth_stats"]["max"], record["depth"]
            )
            stats["depth_stats"]["sum"] += record["depth"]

            current_shard_batch.append(record)

            if len(current_shard_batch) >= positions_per_shard:
                train_batch, test_samples = sample_and_split_batch(
                    current_shard_batch, samples_per_shard_for_test
                )

                if test_samples:
                    test_set_accumulated.extend(test_samples)
                    stats["test_positions"] += len(test_samples)

                # Write training shard
                shard_file_path = output_dir_path / f"shard_{shard_index}.parquet"
                written = write_batch_to_parquet(
                    train_batch, shard_file_path, ARROW_SCHEMA
                )
                stats["train_positions"] += written
                logger.info(f"Written training shard {shard_index} ({written} records)")

                current_shard_batch.clear()
                shard_index += 1

    # Process final batch
    if current_shard_batch:
        train_batch, test_samples = sample_and_split_batch(
            current_shard_batch, samples_per_shard_for_test
        )

        if test_samples:
            test_set_accumulated.extend(test_samples)
            stats["test_positions"] += len(test_samples)

        # Write final training shard
        shard_file_path = output_dir_path / f"shard_{shard_index}.parquet"
        written = write_batch_to_parquet(train_batch, shard_file_path, ARROW_SCHEMA)
        stats["train_positions"] += written
        logger.info(f"Written final training shard {shard_index} ({written} records)")

    # Write test set
    if test_set_accumulated:
        test_written = write_batch_to_parquet(
            test_set_accumulated, test_output_path, ARROW_SCHEMA
        )
        logger.info(
            f"Written test set with {test_written} records to {test_output_path}"
        )
    else:
        logger.warning("No samples were collected for the test set.")

    # Calculate average depth
    avg_depth = (
        stats["depth_stats"]["sum"] / stats["total_positions"]
        if stats["total_positions"] > 0
        else 0
    )

    # Prepare metadata
    metadata = {
        "total_positions_processed": stats["total_positions"],
        "train_set_size": stats["train_positions"],
        "test_set_size": stats["test_positions"],
        "depth_stats": {
            "min": stats["depth_stats"]["min"],
            "max": stats["depth_stats"]["max"],
            "avg": round(avg_depth, 2),
        },
        "min_depth_threshold": min_depth_threshold,
        "positions_per_shard": positions_per_shard,
        "samples_per_shard_for_test": samples_per_shard_for_test,
    }

    metadata_path = output_dir_path / "metadata.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata file: {e}")

    logger.info("--- Processing Finished ---")
    logger.info(f"Final training shards written to: {output_dir_path}")
    logger.info(f"Test set written to: {test_output_path}")


if __name__ == "__main__":
    try:
        generate()
    except Exception as main_e:
        logger.critical(f"Critical error in main execution: {main_e}", exc_info=True)
        sys.exit(1)
