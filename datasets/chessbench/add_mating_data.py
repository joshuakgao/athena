import glob
import os

import chess
import chess.engine
from tqdm import tqdm

from datasets.chessbench.utils import constants
from datasets.chessbench.utils.bagz import BagReader, BagWriter

engine_path = (
    "/home/jkgao/athena/models/stockfish"  # Adjust path to your Stockfish binary
)


def add_mate_annotations(input_bag, output_bag):
    reader = BagReader(input_bag)
    writer = BagWriter(output_bag)

    # Use tqdm with total number of records for more accurate progress
    total_records = len(reader) if hasattr(reader, "__len__") else None

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        with tqdm(
            total=total_records,
            unit="record",
            desc=f"Annotating {os.path.basename(input_bag)}",
        ) as pbar:
            for record in reader:
                fen, move, win_prob = constants.CODERS["action_value"].decode(record)
                mate = 0
                if win_prob in (1.0, 0.0):
                    board = chess.Board(fen)
                    board.push(chess.Move.from_uci(move))
                    info = engine.analyse(board, chess.engine.Limit(time=0.05))
                    score = info.get("score")
                    if score is not None and score.is_mate():
                        mate = score.relative.mate() * -1
                        # print(
                        #     f"Fen: {fen}, Move: {move}, Win Prob: {win_prob}, Mate: {mate}"
                        # )

                new_record = constants.CODERS["action_value_with_mate"].encode(
                    (fen, move, win_prob, mate)
                )
                writer.write(new_record)
                pbar.update(1)

    writer.close()


def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bag_files = glob.glob(os.path.join(input_dir, "*.bag"))
    if not bag_files:
        print(f"No .bag files found in {input_dir}")
        return
    with tqdm(bag_files, desc="Processing .bag files", total=len(bag_files)) as pbar:
        for bag_file in pbar:
            base = os.path.basename(bag_file)
            output_bag = os.path.join(output_dir, base)
            pbar.set_postfix(file=base)
            add_mate_annotations(bag_file, output_bag)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Annotate all .bag files in a directory with mate-in-N info."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/chessbench/data/train",
        help="Directory containing .bag files (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/chessbench/data/train_with_mate",
        help="Directory to write annotated .bag files (default: %(default)s)",
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
