"""
Annotate a .bag file that stores (FEN, move, win-probability) triples
with mate-in-N information produced by Stockfish.

Mate labels:
    "#"   – the move itself gives immediate checkmate
    +N    – mover can force mate in N plies
    -N    – mover will be mated in N plies
    "-"   – no forced mate detected within engine depth / time limit
"""

import argparse
import os

import chess
import chess.engine
from tqdm import tqdm

from datasets.chessbench.utils import constants
from datasets.chessbench.utils.bagz import BagReader, BagWriter

ENGINE_PATH = "models/stockfish"  # adjust if necessary
ENGINE_LIMIT = chess.engine.Limit(time=0.05)  # 50 ms per position


def add_mate_annotations(
    input_bag: str, output_bag: str, max_datapoints: int | None = None
) -> None:
    """Read `input_bag`, annotate with mate info, and write to `output_bag`."""
    reader = BagReader(input_bag)
    writer = BagWriter(output_bag)

    total_records = len(reader) if hasattr(reader, "__len__") else None
    if max_datapoints is not None:
        total_records = (
            min(total_records, max_datapoints)
            if total_records is not None
            else max_datapoints
        )

    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine, tqdm(
        total=total_records,
        unit="record",
        desc=f"Annotating {os.path.basename(input_bag)}",
    ) as pbar:

        for idx, record in enumerate(reader):
            if max_datapoints is not None and idx >= max_datapoints:
                break

            fen, move_str, win_prob = constants.CODERS["action_value"].decode(record)
            board = chess.Board(fen)
            mover = board.turn  # colour that made `move_str`
            move = chess.Move.from_uci(move_str)
            board.push(move)

            mate_label: str | int = "-"

            # Immediate mate by the move itself
            if board.is_checkmate():
                mate_label = "#"

            # Otherwise let Stockfish evaluate
            elif win_prob in (1.0, 0.0):
                info = engine.analyse(board, ENGINE_LIMIT)
                score = info.get("score")
                if score is not None and score.is_mate():
                    # Re-orient the score to the player who just moved
                    mate_label = score.pov(mover).mate()  # ±N in plies

            # Encode and write the new record
            new_record = constants.CODERS["action_value_with_mate"].encode(
                (fen, move_str, win_prob, mate_label)
            )
            writer.write(new_record)
            pbar.update(1)

    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Annotate a .bag file with mate-in-N information."
    )
    parser.add_argument("--input_bag", required=True, help="Path to input .bag file")
    parser.add_argument(
        "--output_bag", required=True, help="Path to output annotated .bag file"
    )
    parser.add_argument(
        "--max_datapoints",
        type=int,
        default=None,
        help="Maximum number of positions to analyse",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_bag), exist_ok=True)

    add_mate_annotations(
        args.input_bag, args.output_bag, max_datapoints=args.max_datapoints
    )


if __name__ == "__main__":
    main()
