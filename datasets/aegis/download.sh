cd /opt/miniconda3/bin
source activate
conda activate athena

# CCRL Games
wget https://computerchess.org.uk/ccrl/4040/CCRL-4040-commented.[2078348].pgn.7z -P datasets/aegis/raw_data/
7z x datasets/aegis/raw_data/CCRL-4040-commented.\[2078348\].pgn.7z -odatasets/aegis/raw_data/
rm datasets/aegis/raw_data/CCRL-4040-commented.\[2078348\].pgn.7z
mv datasets/aegis/raw_data/CCRL-4040-commented.\[2078348\].pgn datasets/aegis/raw_data/ccrl4040-commented.pgn

wget https://computerchess.org.uk/ccrl/Chess324/CCRL-Chess324-commented.[66600].pgn.7z -P datasets/aegis/raw_data/
7z x datasets/aegis/raw_data/CCRL-Chess324-commented.\[66600\].pgn.7z -odatasets/aegis/raw_data/
rm datasets/aegis/raw_data/CCRL-Chess324-commented.\[66600\].pgn.7z
mv datasets/aegis/raw_data/CCRL-Chess324-commented.\[66600\].pgn datasets/aegis/raw_data/ccrlchess324-commented.pgn

# Leela Selfplay Games
curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-1.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-1

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-2.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-2

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-3.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-3

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-4.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-4

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-5.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-5

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-6.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-6\

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-7.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-7

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-8.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-8

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-9.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-9

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-10.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-10

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-11.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-11

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-12.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-12

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-13.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-13

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-14.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-14

curl -L -o datasets/aegis/raw_data/leela-chess-zero-self-play-chess-games-dataset-15.zip\
  https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-15

download_dir="datasets/aegis/raw_data"
for file in "$download_dir"/*.zip; do
  echo "Unzipping $file..."
  unzip -o "$file" -d "$download_dir"
  rm "$file"
done

# Stockfish Selfplay Games
gdown 1qrNvGVW50aTqkkH8l2GZOzV8fVmT2AzI -O datasets/aegis/raw_data/stockfish_games_1.pgn.gz
gdown 1waF1Da9KyNAsj1tb55FDEi0LffPJK4Qa -O datasets/aegis/raw_data/stockfish_games_2.pgn.gz
gdown 17j1U2mJ8dBf6O8FE5yhEhUmpUJtfIjwn -O datasets/aegis/raw_data/stockfish_games_3.pgn.gz
gunzip datasets/aegis/raw_data/stockfish_games_1.pgn.gz
gunzip datasets/aegis/raw_data/stockfish_games_2.pgn.gz
gunzip datasets/aegis/raw_data/stockfish_games_3.pgn.gz

# Lichess Evaluated Positions
wget https://database.lichess.org/lichess_db_eval.jsonl.zst -P datasets/aegis/raw_data/
pzstd -d datasets/aegis/raw_data/lichess_db_eval.jsonl.zst
rm datasets/aegis/raw_data_/lichess_db_eval.jsonl.zst