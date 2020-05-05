#!/bin/bash

#SBATCH --partition=htc-devel
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=offline_training_devel

module purge
module load python/anaconda3/2019.03
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.5__cuda-10.1

epochs=1000
shard_img_size=256
final_size=128
batch_size=64  # about 100s per epoch with these settings, trains a fair bit even on devel K40 :)
shard_dir=$DATA/repos/zoobot/data/decals/shards/multilabel_master_filtered_$shard_img_size

$DATA/envs/zoobot/bin/python offline_training.py --experiment-dir $DATA/repos/zoobot/results/latest_offline_devel --shard-img-size $shard_img_size --train-dir $shard_dir/train --eval-dir $shard_dir/eval --epochs $epochs --batch-size $batch_size --final-size $final_size  
