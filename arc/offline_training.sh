#!/bin/bash

#SBATCH --partition=htc
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:55:00
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=offline_training_x1

module purge
module load python/anaconda3/2019.03
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.5__cuda-10.1

epochs=1000
batch_size=128  # fits on V100, not my laptop...
shard_img_size=424
final_size=224
shard_dir=$DATA/repos/zoobot/data/gz2/shards/all_featp5_facep5_424_arc
# shard_dir=$DATA/repos/zoobot/data/decals/shards/multilabel_master_filtered_$shard_img_size
# shard_dir=$DATA/repos/zoobot/data/gz2/shards/multilabel_master_filtered_$shard_img_size


echo $epochs $batch_size $shard_img_size $final_size $shard_dir

$DATA/envs/zoobot/bin/python offline_training.py --experiment-dir $DATA/repos/zoobot/results/latest_offline_full --shard-img-size $shard_img_size --train-dir $shard_dir/train_shards --eval-dir $shard_dir/eval_shards --epochs $epochs --batch-size $batch_size --final-size $final_size  
