#!/bin/bash

#SBATCH --partition=htc
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:50:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12288
#SBATCH --job-name=continue_simulation

module purge
module load python/anaconda3/2019.03
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.5__cuda-10.1

# source activate $DATA/envs/zoobot  # only works in THIS script
export PYTHON=$DATA/envs/zoobot/bin/python

# catalog_dir=data/decals/prepared_catalogs/decals_multiq
# shard_dir=data/decals/shards/decals_multiq_128_sim_init_2500_featp4

# and switch label cols in make_shards.py, create_instructions.py, and run_iteration.py, and version in iterations.py, for now
catalog_dir=data/gz2/prepared_catalogs/all_featp5_facep5_arc
shard_dir=data/gz2/shards/all_featp5_facep5_sim_256_arc

experiment_dir=data/experiments/live/latest
instructions_dir=$experiment_dir/instructions
this_iteration_dir=$experiment_dir/iteration_1
previous_iteration_dir=$experiment_dir/iteration_0
options='baseline_test'

$PYTHON zoobot/active_learning/run_iteration.py --instructions-dir data/experiments/decals_multiq_sim/instructions --this-iteration-dir $this_iteration_dir --previous-iteration-dir $previous_iteration_dir --options $options