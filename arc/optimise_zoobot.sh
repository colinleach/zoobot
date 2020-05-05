#!/bin/bash

#SBATCH --partition=htc
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=optimise_zoobot

module purge
module load python/anaconda3/2019.03
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.5__cuda-10.1

ls $LD_LIBRARY_PATH

nvidia-smi

# research code:
$DATA/envs/zoobot/bin/python $DATA/repos/zoobot/optimise_zoobot.py

