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
module load gpu/cudnn/7.5.0__cuda-10.0

# minimal Tensorflow example:

echo 'Running TF 2.1'
# with my env (TF 2.1), cannot load CUDA without cudnnn >7.6.4
$DATA/envs/zoobot/bin/python $DATA/repos/zoobot/minimal_tensorflow.py

echo 'Running alternative env'
# Yassamine's env only has TF 1.10
$DATA/tensor-env/bin/python $DATA/repos/zoobot/minimal_tensorflow.py
