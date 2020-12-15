#!/bin/bash

#SBATCH --partition=htc-devel
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=predict-devel

## SBATCH --partition=htc
## SBATCH --ntasks-per-node=1
## SBATCH --time=23:55:00
## SBATCH --gres=gpu:v100:1
## SBATCH --job-name=predict

module purge
module load python/anaconda3/2019.03
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.5__cuda-10.1

$DATA/envs/zoobot/bin/python predictions_on_images.py
