import os
import chess
import chess.engine
from tqdm import tqdm

from datasets.chessbench.utils import constants
from datasets.chessbench.utils.bagz import BagReader, BagWriter

engine_path = "models/stockfish"


def add_mate_annotations(input_bag, output_bag, max_datapoints=None):
    reader = BagReader(input_bag)
    writer = BagWriter(output_bag)

    total_records = len(reader) if hasattr(reader, "__len__") else None
    if max_datapoints is not None:
        total_records = (
            min(total_records, max_datapoints)
            if total_records is not None
            else max_datapoints
        )

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        with tqdm(
            total=total_records,
            unit="record",
            desc=f"Annotating {os.path.basename(input_bag)}",
        ) as pbar:
            for idx, record in enumerate(reader):
                if max_datapoints is not None and idx >= max_datapoints:
                    break
                fen, move, win_prob = constants.CODERS["action_value"].decode(record)
                mate = 0
                if win_prob in (1.0, 0.0):
                    board = chess.Board(fen)
                    board.push(chess.Move.from_uci(move))
                    info = engine.analyse(board, chess.engine.Limit(time=0.05))
                    score = info.get("score")
                    if score is not None and score.is_mate():
                        mate = score.relative.mate() * -1

                new_record = constants.CODERS["action_value_with_mate"].encode(
                    (fen, move, win_prob, mate)
                )
                writer.write(new_record)
                pbar.update(1)

    writer.close()


def main(input_bag, output_bag, max_datapoints=None):
    os.makedirs(os.path.dirname(output_bag), exist_ok=True)
    add_mate_annotations(input_bag, output_bag, max_datapoints=max_datapoints)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Annotate a single .bag file with mate-in-N info."
    )
    parser.add_argument(
        "--input_bag",
        type=str,
        required=True,
        help="Path to input .bag file",
    )
    parser.add_argument(
        "--output_bag",
        type=str,
        required=True,
        help="Path to output annotated .bag file",
    )
    parser.add_argument(
        "--max_datapoints",
        type=int,
        default=None,
        help="Maximum number of datapoints to process",
    )
    args = parser.parse_args()

    main(args.input_bag, args.output_bag, max_datapoints=args.max_datapoints)
