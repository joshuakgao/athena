import os
import chess.pgn
from tqdm import tqdm
from utils.chess_utils import is_fen_valid, is_fen_end_of_game
from utils.rate_tracker import RateTracker


def extract_fens_from_pgn(pgn_path, output_path):
    """
    Extracts FENs from a single PGN file and saves them to a text file.

    Args:
        pgn_path (str): Path to the input PGN file.
        output_path (str): Path to the output text file.
    """
    pgn_file = open(pgn_path)
    fens = set()  # Use a set to avoid duplicates

    i = 0
    rate_tracker = RateTracker(unit="games")
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break  # End of file

        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            fen = board.fen()

            if is_fen_valid(fen) and not is_fen_end_of_game(fen):
                fens.add(fen)

        i += 1
        rate_tracker.increment()
        if i % 1000 == 0:
            rate_tracker.log_rate()

    # Save FENs to a file
    with open(output_path, "w") as f:
        for fen in fens:
            f.write(fen + "\n")


def process_all_pgns(input_dir, output_dir):
    """
    Processes all .pgn files in the input directory and saves FENs to the output directory.

    Args:
        input_dir (str): Directory containing the .pgn files.
        output_dir (str): Directory to save the FEN text files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all .pgn files in the input directory
    pgn_files = [f for f in os.listdir(input_dir) if f.endswith(".pgn")]

    for pgn_file in pgn_files:
        # Construct the output file path
        output_file = os.path.splitext(pgn_file)[0] + "_fens.txt"
        output_path = os.path.join(output_dir, output_file)

        # Skip if the output file already exists
        if os.path.exists(output_path):
            print(f"Skipping {pgn_file} (output already exists: {output_file})")
            continue

        # Extract FENs and save to output file
        pgn_path = os.path.join(input_dir, pgn_file)
        print(f"Processing {pgn_file} -> {output_file}...")
        extract_fens_from_pgn(pgn_path, output_path)
        print(f"Processed {pgn_file} -> {output_file}")


# Example usage
input_directory = "datasets/lichess/data"
output_directory = "datasets/lichess/data"
process_all_pgns(input_directory, output_directory)
