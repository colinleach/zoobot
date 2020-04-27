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

source activate $DATA/envs/zoobot

# minimal example, works but profiler cannot load without CUPTI
$DATA/envs/zoobot/bin/python $DATA/repos/zoobot/minimal_tensorflow.py

catalog_dir=data/decals/prepared_catalogs/decals_multiq
shard_dir=data/decals/shards/decals_multiq_128_sim_init_2500_featp4
experiment_dir=data/experiments/live/latest
instructions_dir=$experiment_dir/instructions
n_iterations=5
baseline='--baseline'
test_flag='--test'

dvc run -d $shard_dir -d $catalog_dir -d production/create_instructions.sh -o $instructions_dir -f $experiment_dir.dvc ./production/create_instructions.sh $catalog_dir $shard_dir $experiment_dir
dvc run --ignore-build-cache -d $shard_dir -d $catalog_dir -d production/run_simulation.sh -o $experiment_dir -f $experiment_dir.dvc ./production/run_simulation.sh $n_iterations $catalog_dir $shard_dir $experiment_dir $baseline $test_flag ''