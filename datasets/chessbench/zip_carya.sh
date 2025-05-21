#!/bin/bash
#SBATCH -J zip
#SBATCH --cpus-per-task=32
#SBATCH -t 99:0:0
#SBATCH --mem-per-cpu=4GB

cd /project/hoskere/jkgao/athena/datasets/chessbench
tar -cf - data_mate/ | pigz -p 32 > data_mate.tar.gz
