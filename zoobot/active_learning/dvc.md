
## Locally

**Run new reduction (optional)**

*Check get_latest.py carefully first to make sure you're not deleting anything!*

`dvc run -o data/decals/classifications/classifications.csv python -f data/decals/classifications.csv.dvc ../gzreduction/get_latest.py`

`dvc push -r s3 data/decals/classifications.dvc`

## EC2

**Create master catalog**

`dvc run -o data/decals/decals_master_catalog.csv -f data/decals/decals_master_catalog.csv.dvc -d zoobot/active_learning/prepare_catalogs.py python zoobot/active_learning/prepare_catalogs.py`

**Specify what you'd like to do**

`master_catalog=data/decals/decals_master_catalog.csv`

`question=bar`

`catalog_dir=data/decals/prepared_catalogs/decals_weak_bars_launch`

`shard_dir=data/decals/shards/decals_weak_bars_launch`

**Define experiment (create experiment catalogs)**

`dvc run -d $master_catalog -d zoobot/active_learning/define_experiment.py -o $catalog_dir -f $catalog_dir.dvc python zoobot/active_learning/define_experiment.py --master-catalog=$master_catalog --question=$question --save-dir=$catalog_dir`

**Create shards**

`dvc run -d $catalog_dir -d zoobot/active_learning/make_shards.py -o $shard_dir -f $shard_dir.dvc python zoobot/active_learning/make_shards.py --labelled-catalog=$catalog_dir/labelled_catalog.csv --unlabelled-catalog=$catalog_dir/unlabelled_catalog.csv --eval-size=5000 --shard-dir=$shard_dir`