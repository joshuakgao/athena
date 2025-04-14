#!/bin/bash
#SBATCH -J athena
#SBATCH -o athena.o%j
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH -t 5-96:0:0
#SBATCH --mem-per-cpu=64GB

# FOR CARYA:
# module add Miniforge3/py3.10
# module add cudatoolkit/12.4
# source activate /project/hoskere/jkgao/.conda/envs/athena

# export CONDA_PKGS_DIRS=/project/hoskere/jkgao/.conda/conda_pkgs_dir/
# export XDG_CACHE_HOME=/project/hoskere/jkgao/.conda/cache/
# export PYTHONPATH=/project/hoskere/jkgao/.conda/envs/athena/bin/python

# free -h
# /project/hoskere/jkgao/.conda/envs/athena/bin/python -m datasets.aegis.generate

# FOR SAIL:
cd /opt/miniconda3/bin
source activate
conda activate athena

cd ~/athena
free -h
~/.conda/envs/athena/bin/python datasets/aegis/generate.py