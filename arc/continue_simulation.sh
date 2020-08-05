#!/bin/bash

#SBATCH --partition=htc
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:50:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12288
#SBATCH --job-name=continue_simulation

module purge
module load python/anaconda3/2019.03
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.5__cuda-10.1

# source activate $DATA/envs/zoobot  # only works in THIS script
export PYTHON=$DATA/envs/zoobot/bin/python

experiment_dir=data/experiments/live/latest_dirichlet_unfiltered_active_m3_warm
instructions_dir=$experiment_dir/instructions
this_iteration_dir=$experiment_dir/iteration_3
previous_iteration_dir=$experiment_dir/iteration_2
options='active'

$PYTHON zoobot/active_learning/run_iteration.py --instructions-dir $instructions_dir --this-iteration-dir $this_iteration_dir --previous-iteration-dir $previous_iteration_dir --options $options
