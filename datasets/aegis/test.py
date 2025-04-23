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
        df_head = df.head()
        for _, row in df_head.iterrows():
            print(row.to_dict())

        for name, fen, history in tests:
            matches = df[(df["fen"] == fen)]
            for _, row in matches.iterrows():
                # print(row.to_dict())
                if list(row["history"]) == history:
                    test_results[name] += 1

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
            [],
        ),
        (
            "King's pawn game",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            [],
            # [
            #     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            #     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # ],
        ),
        (
            "Queen's pawn game",
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
            [],
            # [
            #     "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
            #     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # ],
        ),
        (
            "Sicilian",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            [],
            # [
            #     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            #     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # ],
        ),
        (
            "Petrov",
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            [],
            # [
            #     "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",
            #     "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            #     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            #     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # ],
        ),
        (
            "Evan's Gambit",
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R b KQkq - 0 1",
            [],
            # [
            #     "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            #     "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            #     "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            #     "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",
            #     "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            #     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            #     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # ],
        ),
        (
            "Ruy Lopez",
            "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            [],
            # [
            #     "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            #     "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            #     "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            #     "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",
            #     "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            #     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            #     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # ],
        ),
        (
            "Mid Game",
            "1B6/Q3n2k/5p2/b2p3p/2pP2b1/p1P5/P3qPPK/5N2 b - - 0 1",
            [],
            # [
            #     "1r6/Q3n2k/3B1p2/b2p3p/2pP2b1/p1P5/P3qPPK/5N2 w - - 0 1",
            #     "1R6/Q3n2k/3B1p2/b2p3p/2pP2b1/p1P5/Pr2qPPK/5N2 b - - 0 1",
            #     "1R6/1Q2n2k/3B1p2/b2p3p/2pP2b1/p1P5/Pr2qPPK/5N2 w - - 0 1",
            #     "1R6/1Q2n2k/3B1p2/b2p3p/2pP2b1/p1P5/P1r1qPPK/5N2 b - - 0 1",
            #     "1R6/1Q2n2k/3B1p2/b2p3p/2pP2b1/p1P5/P1r1qPP1/5NK1 w - - 0 1",
            #     "1R6/1Q5k/3B1pn1/b2p3p/2pP2b1/p1P5/P1r1qPP1/5NK1 b - - 0 1",
            #     "1R6/7k/3B1pn1/b2p3p/2pP2b1/p1P5/P1r1qPP1/1Q3NK1 w - - 0 1",
            # ],
        ),
    ]

    validate(tests)
