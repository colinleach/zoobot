#!/bin/bash

#SBATCH --partition=htc-devel
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12288
#SBATCH --job-name=submit_simulation_devel

module purge
module load python/anaconda3/2019.03
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.5.0__cuda-10.0

# _____module load gpu/cudnn/7.5.0__cuda-10.0
# fails with: Loaded runtime CuDNN library: 7.5.0 but source was compiled with: 7.6.4.  CuDNN library major and minor version needs to match or have higher minor version in case of CuDNN 7.0 or later version. If using a binary install (I am), upgrade your CuDNN library.

# ____ module load gpu/cudnn/7.6.0__cuda-9.0
# fails with: Could not load dynamic library 'libcudnn.so.7'; dlerror: libcudnn.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /system/software/arcus-htc/cuda-toolkit/10.1.243/lib64:/system/software/arcus-htc/cuda-toolkit/10.1.243/lib


# source activate $DATA/envs/zoobot  # only works in THIS script
export PYTHON=$DATA/envs/zoobot/bin/python

catalog_dir=data/decals/prepared_catalogs/decals_multiq
shard_dir=data/decals/shards/decals_multiq_128_sim_init_2500_featp4
experiment_dir=data/experiments/live/latest
instructions_dir=$experiment_dir/instructions
n_iterations=5
baseline='--baseline'
test_flag='--test'

./production/create_instructions.sh $catalog_dir $shard_dir $experiment_dir
./production/run_simulation.sh $n_iterations $catalog_dir $shard_dir $experiment_dir $baseline $test_flag ''

# dvc run -d $shard_dir -d $catalog_dir -d production/create_instructions.sh -o $instructions_dir -f $experiment_dir.dvc ./production/create_instructions.sh $catalog_dir $shard_dir $experiment_dir
# dvc run --ignore-build-cache -d $shard_dir -d $catalog_dir -d production/run_simulation.sh -o $experiment_dir -f $experiment_dir.dvc ./production/run_simulation.sh $n_iterations $catalog_dir $shard_dir $experiment_dir $baseline $test_flag ''