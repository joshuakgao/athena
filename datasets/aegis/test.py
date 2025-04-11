import concurrent.futures
import os
import sys
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.logger import logger

output_dir_path = Path("datasets/aegis/data")


def validate_file(parquet_file, fen, history):
    """Helper function to validate a single parquet file"""
    try:
        df = pq.read_table(parquet_file).to_pandas()

        fen_matches = df[df["history"].apply(len) < 7]
        if not fen_matches.empty:
            df0 = fen_matches.iloc[0]
            print("First FEN match:", df0["fen"])
            print("First history option:", df0["history"])
            print("Bot:", df0["bot"])
            print("Elo:", df0["elo"])
        for _, row in fen_matches.iterrows():
            if list(row["history"]) == history:
                return 1  # Found a match
    except Exception as e:
        logger.error(f"Error processing {parquet_file}: {e}")
    return 0  # No match found


def validate(fen, history, max_cores=32):
    """Multi-threaded validation function"""
    workers = min(os.cpu_count(), max_cores)
    logger.info(f"Using {workers} CPU cores.")

    aegis = sorted(output_dir_path.glob("*.parquet"))
    count = 0

    # Create a partial function with fixed fen and history arguments
    validate_func = partial(validate_file, fen=fen, history=history)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Process files in parallel
        results = list(tqdm(executor.map(validate_func, aegis), total=len(aegis)))
        count = sum(results)

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
    tests = [
        # (
        #     "1B1b1k1r/1Q3pp1/1p1Np3/8/1P4p1/5qP1/5P2/R4RK1 w - - 0 1",
        #     [
        #         "1B1b3r/1Q2kpp1/1p1Np3/8/1P4p1/5qP1/5P2/R4RK1 b - - 0 1",
        #         "1BQb3r/4kpp1/1p1Np3/8/1P4p1/5qP1/5P2/R4RK1 w - - 0 1",
        #         "1BQbk2r/5pp1/1p1Np3/8/1P4p1/5qP1/5P2/R4RK1 b k - 0 1",
        #         "1BQbk2r/5pp1/1p2p3/1N6/1P4p1/5qP1/5P2/R4RK1 w k - 0 1",
        #         "1BQ1k2r/4bpp1/1p2p3/1N6/1P4p1/5qP1/5P2/R4RK1 b k - 0 1",
        #         "1Bn1k2r/4bpp1/Qp2p3/1N6/1P4p1/5qP1/5P2/R4RK1 w k - 0 1",
        #         "1Bn1k2r/4bpp1/Qp2pq2/1N6/1P4p1/5NP1/5P2/R4RK1 b k - 0 1",
        #     ],
        # ),
        # (
        #     "1Q3nk1/p2r2p1/5p1p/8/PPR2N1P/3p2P1/5PK1/1q6 b - - 0 1",
        #     [
        #         "5nk1/p1Qr2p1/5p1p/8/PPR2N1P/3p2P1/5PK1/1q6 w - - 0 1",
        #         "3r1nk1/p1Q3p1/5p1p/8/PPR2N1P/3p2P1/5PK1/1q6 b - - 0 1",
        #         "3r1nk1/p5p1/2Q2p1p/8/PPR2N1P/3p2P1/5PK1/1q6 w - - 0 1",
        #         "3r1nk1/p5p1/2Q2p1p/8/PPRp1N1P/6P1/5PK1/1q6 b - - 0 1",
        #         "3r1nk1/p5p1/2Q2p1p/8/PPRp3P/3N2P1/5PK1/1q6 w - - 0 1",
        #         "3r1nk1/p5p1/2Q2p1p/8/PPRp3P/3N2P1/q4PK1/8 b - - 0 1",
        #         "3r1nk1/p5p1/2Q2p1p/4N3/PPRp3P/6P1/q4PK1/8 w - - 0 1",
        #     ],
        # ),
        # (
        #     "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",
        #     [
        #         "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",
        #         "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        #         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        #         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        #     ],
        # ),
        (
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
            [
                "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
                "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1",
                "rnbqkbnr/ppp1pppp/8/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 0 1",
                "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ],
        ),
        (
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            [
                "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
                "r1bqkbnr/pp1ppppp/8/2p5/3nP3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",
                "r1bqkbnr/pp1ppppp/8/1Bp5/3nP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
                "r1bqkbnr/pp1ppppp/2n5/1Bp5/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
                "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
                "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",
            ],
        ),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", []),
    ]

    n_tests = len(tests)
    n_tests_passed = 0
    for test in tests:
        fen, history = test
        passed = validate(fen, history, max_cores=1)
        if passed:
            n_tests_passed += 1
        else:
            print("TEST FAILED:")
            print(fen)
            print(history)
    print(f"Passed {n_tests_passed} out of {n_tests}.")
