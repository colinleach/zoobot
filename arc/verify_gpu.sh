#!/bin/bash

#SBATCH --partition=htc-devel
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=debug

module purge
module load python/anaconda3/2019.03
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.0__cuda-9.0

source activate /data/phys-zooniverse/chri5177/envs/zoobot

python /data/phys-zooniverse/chri5177/repos/zoobot/verify_gpu.py
