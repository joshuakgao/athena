import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.logger import logger

data_dir = Path("datasets/aegis/data")


def validate(tests):
    aegis = sorted(data_dir.glob("*.parquet"))
    test_results = {test[0]: 0 for test in tests}  # Track counts per test

    for parquet_file in tqdm(aegis, desc="Validating files"):
        df = pq.read_table(parquet_file).to_pandas()

        # Print for debugging
        df_head = df.head(n=20)
        for _, row in df_head.iterrows():
            print(row.to_dict())

        for name, fen in tests:
            matches = df[(df["fen"] == fen)]
            test_results[name] += len(matches)

    # Return validation result
    for name, count in test_results.items():
        if count == 1:
            logger.info(f"{name}: passed ✅")
        elif count == 0:
            logger.info(f"{name}: None positions found ❓")
        else:
            logger.info(f"{name}: Duplicate positions found ❌")


if __name__ == "__main__":
    tests = [
        (
            "Starting Position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "King's pawn game",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "Queen's pawn game",
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "Sicilian",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "Petrov",
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        ),
        (
            "Evan's Gambit",
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R b KQkq - 0 1",
        ),
        (
            "Ruy Lopez",
            "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        ),
        (
            "Mid Game",
            "1B6/Q3n2k/5p2/b2p3p/2pP2b1/p1P5/P3qPPK/5N2 b - - 0 1",
        ),
    ]

    validate(tests)
