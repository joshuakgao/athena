#!/bin/bash
#SBATCH -J athena
#SBATCH -o athena.o%j
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gpus=1

# FOR CARYA:
# module add Miniforge3/py3.10
# module add cudatoolkit/12.4
# source activate /project/hoskere/jkgao/.conda/envs/athena

# export CONDA_PKGS_DIRS=/project/hoskere/jkgao/.conda/conda_pkgs_dir/
# export XDG_CACHE_HOME=/project/hoskere/jkgao/.conda/cache/
# export PYTHONPATH=/project/hoskere/jkgao/.conda/envs/athena/bin/python

# free -h
# /project/hoskere/jkgao/.conda/envs/athena/bin/python train.py

# FOR SAIL:
cd /opt/miniconda3/bin
source activate
conda activate athena

cd ~/athena
free -h
~/.conda/envs/athena/bin/python train.py