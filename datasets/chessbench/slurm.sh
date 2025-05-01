#!/bin/bash
#SBATCH -J bash
#SBATCH -o bash.o%j
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

bash datasets/chessbench/download.sh