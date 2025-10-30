#!/bin/bash
#SBATCH -J bash
#SBATCH -o bash.o%j
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gpus=2

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
free -h
df -h
# Launch multi-GPU training using torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    src/athena/train_multi_gpu.py