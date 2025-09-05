"""Annotate a .bag file that stores (FEN, move, win-probability) triples with mate-in-N information produced by Stockfish.

Mate labels:
    "#"   – the move itself gives immediate checkmate
    +N    – mover can force mate in N plies
    -N    – mover will be mated in N plies
    "-"   – no forced mate detected within engine depth / time limit
"""

import argparse
import os
from multiprocessing import Pool, cpu_count

import chess
import chess.engine
from tqdm import tqdm

from athena.datasets.chessbenchmate.utils.bagz import BagReader, BagWriter
from athena.datasets.chessbenchmate.utils.constants import CODERS

ENGINE_PATH = "models/stockfish"
ENGINE_LIMIT = chess.engine.Limit(time=0.05)


def annotate_single_record(record: bytes) -> bytes:
    """Annotate a single record with mate-in-N information."""
    fen, move_str, win_prob = CODERS["action_value"].decode(record)
    board = chess.Board(fen)
    mover = board.turn
    move = chess.Move.from_uci(move_str)
    board.push(move)

    mate_label: str | int

    if board.is_checkmate():
        mate_label = "#"
    elif win_prob in (1.0, 0.0):
        with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
            info = engine.analyse(board, ENGINE_LIMIT)
            score = info.get("score")
            if score and score.is_mate():
                mate = score.pov(mover).mate()
                mate_label = mate if mate is not None else "-"
            else:
                mate_label = "-"
    else:
        mate_label = "-"

    return CODERS["action_value_with_mate"].encode((fen, move_str, win_prob, mate_label))


def add_mate_annotations(
    input_bag: str, output_bag: str, max_datapoints: int | None = None
) -> None:
    """Annotate a .bag file with mate-in-N information."""
    reader = BagReader(input_bag)
    writer = BagWriter(output_bag)

    records = list(reader)[:max_datapoints] if max_datapoints else list(reader)

    print(cpu_count(), flush=True)
    with Pool(processes=cpu_count()) as pool:
        for annotated_record in tqdm(
            pool.imap(annotate_single_record, records),
            total=len(records),
            unit="record",
        ):
            writer.write(annotated_record)

    writer.close()


def main():
    """Main entry point to parse command line arguments to add mating data to chessbench."""
    parser = argparse.ArgumentParser(description="Annotate a .bag file with mate-in-N information.")
    parser.add_argument("--input_bag", required=True, help="Path to input .bag file")
    parser.add_argument("--output_bag", required=True, help="Path to output annotated .bag file")
    parser.add_argument(
        "--max_datapoints",
        type=int,
        default=None,
        help="Maximum number of positions to analyse",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_bag), exist_ok=True)

    add_mate_annotations(args.input_bag, args.output_bag, max_datapoints=args.max_datapoints)


if __name__ == "__main__":
    main()
