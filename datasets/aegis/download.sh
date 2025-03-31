#!/bin/bash

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

for file in datasets/aegis/raw_data/*.zip; do
    unzip -o "$file" -d datasets/aegis/raw_data/
    rm "$file"
done

wget https://www.computerchess.org.uk/ccrl/404/CCRL-404.[1542299].pgn.7z -P datasets/aegis/raw_data/
7z x datasets/aegis/raw_data/CCRL-404.\[1542299\].pgn.7z -odatasets/aegis/raw_data/
rm datasets/aegis/raw_data/CCRL-404.\[1542299\].pgn.7z
mv datasets/aegis/raw_data/CCRL-404.\[1542299\].pgn datasets/aegis/raw_data/ccrl.pgn