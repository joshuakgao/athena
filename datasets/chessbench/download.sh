#!/bin/bash
#SBATCH -J bash
#SBATCH -o bash.o%j
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

cd datasets/chessbench/data

set -ex

wget https://storage.googleapis.com/searchless_chess/data/eco_openings.pgn
wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv

mkdir test
cd test
wget https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag
cd ..

mkdir train
cd train
for idx in $(seq -f "%05g" 0 2147) # range of shards you want to download. Max of 2147. Requires 1.1TB of storage.
do
  wget https://storage.googleapis.com/searchless_chess/data/train/action_value-$idx-of-02148_data.bag
done