wget https://computerchess.org.uk/ccrl/4040/CCRL-4040.[2078348].pgn.7z -P datasets/aegis/raw_data/
7z x datasets/aegis/raw_data/CCRL-4040.\[2078348\].pgn.7z -odatasets/aegis/raw_data/
rm datasets/aegis/raw_data/CCRL-4040.\[2078348\].pgn.7z
mv datasets/aegis/raw_data/CCRL-4040.\[2078348\].pgn datasets/aegis/raw_data/ccrl4040.pgn

wget https://www.computerchess.org.uk/ccrl/404/CCRL-404.[1563236].pgn.7z -P datasets/aegis/raw_data/
7z x datasets/aegis/raw_data/CCRL-404.\[1563236\].pgn.7z -odatasets/aegis/raw_data/
rm datasets/aegis/raw_data/CCRL-404.\[1563236\].pgn.7z
mv datasets/aegis/raw_data/CCRL-404.\[1563236\].pgn datasets/aegis/raw_data/ccrl404.pgn

wget https://computerchess.org.uk/ccrl/404FRC/CCRL-404FRC.[736287].pgn.7z -P datasets/aegis/raw_data/
7z x datasets/aegis/raw_data/CCRL-404FRC.\[736287\].pgn.7z -odatasets/aegis/raw_data/
rm datasets/aegis/raw_data/CCRL-404FRC.\[736287\].pgn.7z
mv datasets/aegis/raw_data/CCRL-404FRC.\[736287\].pgn datasets/aegis/raw_data/ccrl404frc.pgn

wget https://computerchess.org.uk/ccrl/402.archive/CCRL.40-2.Archive.[2165313].pgn.7z -P datasets/aegis/raw_data/
7z x datasets/aegis/raw_data/CCRL.40-2.Archive.\[2165313\].pgn.7z -odatasets/aegis/raw_data/
rm datasets/aegis/raw_data/CCRL.40-2.Archive.\[2165313\].pgn.7z
mv datasets/aegis/raw_data/CCRL.40-2.Archive.\[2165313\].pgn datasets/aegis/raw_data/ccrl40-2archive.pgn

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

# Unzip all downloaded player game zip files
download_dir="datasets/aegis/raw_data"
for file in "$download_dir"/*.zip; do
  echo "Unzipping $file..."
  unzip -o "$file" -d "$download_dir"
  rm "$file"  # Remove the zip file after extraction
done