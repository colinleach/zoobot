#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --job-name=single_core
#SBATCH --ntasks-per-node=1
#SBATCH --partition=htc

module purge

python /data/phys-zooniverse/chri5177/repos/zoobot/make_decals_tfrecords.py --labelled-catalog=/home/walml/repos/zoobot/data/latest_labelled_catalog_256.csv --eval-size=5525 --shard-dir=/data/phys-zooniverse/chri5177/repos/zoobot/data/decals/shards/multilabel_256 --img-size 256
