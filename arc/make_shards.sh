#!/bin/bash

#SBATCH --partition=htc
#SBATCH --ntasks-per-node=1
#SBATCH --time=95:50:00
#SBATCH --job-name=make_shards

module purge
module load python/anaconda3/2019.03

export PYTHON=$DATA/envs/zoobot/bin/python

$PYTHON zoobot/active_learning/make_shards.py --labelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_n2_arc/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_n2_arc/unlabelled_catalog.csv --eval-size 10000 --shard-dir=data/decals/shards/all_2p5_unfiltered_n2  --img-size 300
