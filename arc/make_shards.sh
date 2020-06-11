#!/bin/bash

#SBATCH --partition=htc
#SBATCH --ntasks-per-node=1
#SBATCH --time=95:50:00
#SBATCH --job-name=make_shards

module purge
module load python/anaconda3/2019.03

export PYTHON=$DATA/envs/zoobot/bin/python

$PYTHON zoobot/active_learning/make_shards.py --labelled-catalog=data/gz2/prepared_catalogs/all_2p5_arc_unfiltered/labelled_catalog.csv --unlabelled-catalog=data/gz2/prepared_catalogs/all_2p5_arc_unfiltered/unlabelled_catalog.csv --eval-size 15000 --shard-dir=data/gz2/shards/all_sim_2p5_unfiltered_300 --img-size 300
