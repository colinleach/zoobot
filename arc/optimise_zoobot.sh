#!/bin/bash

#SBATCH --partition=htc-devel
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12288
#SBATCH --job-name=offline_training_devel

module purge
module load python/anaconda3/2019.03
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.5__cuda-10.1

# minimal Tensorflow example:
# $DATA/envs/zoobot/bin/python $DATA/repos/zoobot/minimal_tensorflow.py

# research code:
$DATA/envs/zoobot/bin/python $DATA/repos/zoobot/optimise_zoobot.py

