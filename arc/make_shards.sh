#!/bin/bash

#SBATCH --partition=htc
#SBATCH --ntasks-per-node=1
#SBATCH --time=95:50:00
#SBATCH --job-name=make_shards

module purge
module load python/anaconda3/2019.03

export PYTHON=$DATA/envs/zoobot/bin/python

CATALOG_NAME=decals_dr  # min 3 classifications

$PYTHON zoobot/active_learning/make_shards.py --labelled-catalog=data/decals/prepared_catalogs/$CATALOG_NAME/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/$CATALOG_NAME/unlabelled_catalog.csv --eval-size 49700 --shard-dir=data/decals/shards/${CATALOG_NAME}_full  --img-size 300
# $PYTHON zoobot/active_learning/make_shards.py --labelled-catalog=data/decals/prepared_catalogs/$CATALOG_NAME/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/$CATALOG_NAME/unlabelled_catalog.csv --eval-size 100 --shard-dir=data/decals/shards/$CATALOG_NAME  --img-size 300 --max-labelled 4000
