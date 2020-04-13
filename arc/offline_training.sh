#!/bin/bash

#SBATCH --partition=htc-devel
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:p100:2
#SBATCH --mem=12288
#SBATCH --job-name=offline_training

module purge
module load python/anaconda3/2019.03
# module load gpu/cuda/10.1.243
# module load gpu/cudnn/7.6.0__cuda-9.0
module load gpu/cuda/10.0.130
module load gpu/cudnn/7.5.0__cuda-10.0

# source activate /data/phys-zooniverse/chri5177/envs/zoobot

/data/phys-zooniverse/chri5177/envs/zoobot/bin/python offline_training.py --experiment-dir /data/phys-zooniverse/chri5177/repos/zoobot/results/latest_offline --shard-img-size 256 --train-dir /data/phys-zooniverse/chri5177/repos/zoobot/data/decals/shards/multilabel_256/train --eval-dir /data/phys-zooniverse/chri5177/repos/zoobot/data/decals/shards/multilabel_256/eval --epochs 150 
