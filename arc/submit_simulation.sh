#!/bin/bash

#SBATCH --partition=htc
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:55:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12288
#SBATCH --job-name=submit_simulation

module purge
module load python/anaconda3/2019.03
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.5__cuda-10.1

# source activate $DATA/envs/zoobot  # only works in THIS script
export PYTHON=$DATA/envs/zoobot/bin/python

# catalog_dir=data/decals/prepared_catalogs/decals_multiq
# shard_dir=data/decals/shards/decals_multiq_128_sim_init_2500_featp4

# and switch label cols in make_shards.py, create_instructions.py, and run_iteration.py, and version in iterations.py, for now
catalog_dir=data/gz2/prepared_catalogs/all_2p5_arc_unfiltered
shard_dir=data/gz2/shards/all_actual_sim_2p5_unfiltered_300

experiment_dir=data/experiments/live/gz2_4q_repeat_opt4_warm_baseline
instructions_dir=$experiment_dir/instructions
n_iterations=10
options='baseline'

# ./production/create_instructions.sh $catalog_dir $shard_dir $experiment_dir $options
./production/run_simulation.sh $n_iterations $catalog_dir $shard_dir $experiment_dir $options

# dvc run -d $shard_dir -d $catalog_dir -d production/create_instructions.sh -o $instructions_dir -f $experiment_dir.dvc ./production/create_instructions.sh $catalog_dir $shard_dir $experiment_dir
# dvc run --ignore-build-cache -d $shard_dir -d $catalog_dir -d production/run_simulation.sh -o $experiment_dir -f $experiment_dir.dvc ./production/run_simulation.sh $n_iterations $catalog_dir $shard_dir $experiment_dir $baseline $test_flag ''
