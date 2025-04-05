import csv
import math
from tqdm import tqdm

# Input CSV file and output directory
csv_file = "datasets/aegis/raw_data/GM_games_dataset.csv"
output_dir = "datasets/aegis/raw_data/"
num_files = 20  # Number of files to split into

print("Counting total games...")

# First pass to count total games
with open(csv_file, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    total_games = sum(1 for _ in reader)

games_per_file = math.ceil(total_games / num_files)

# Second pass to write the split files
with open(csv_file, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for file_num in tqdm(range(num_files)):
        output_file = f"{output_dir}GM_games_{file_num+1}.pgn"
        start_idx = file_num * games_per_file
        end_idx = start_idx + games_per_file

        with open(output_file, mode="w", encoding="utf-8") as pgn_file:
            for i, row in enumerate(reader):
                if i >= end_idx:
                    break  # Move to next file
                if i >= start_idx:
                    pgn_data = row["pgn"]
                    pgn_file.write(pgn_data + "\n\n")

                # Reset reader position if we're starting a new file
                if i == end_idx - 1 and file_num < num_files - 1:
                    file.seek(0)  # Rewind to start of file
                    next(reader)  # Skip header row
                    break

print(f"Successfully split {total_games} games into {num_files} files in {output_dir}")
