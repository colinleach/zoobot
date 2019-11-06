
## Locally

**Run new reduction (optional)**

*Check get_latest.py carefully first to make sure you're not deleting anything!*

`dvc run -o data/decals/classifications/classifications.csv python -f data/decals/classifications.csv.dvc ../gzreduction/get_latest.py`

`dvc push -r s3 data/decals/classifications.dvc`

## EC2

<!-- `shard_dir=data/decals/shards/decals_weak_bars_sim` -->
<!-- `question=bar` -->
<!-- `catalog_dir=data/decals/prepared_catalogs/decals_weak_bars_launch`
`shard_dir=data/decals/shards/decals_weak_bars_launch`
`experiment_dir=data/experiments/simulation/decals_weak_bars_launch_test` -->

**Specify what you'd like to do**

`master_catalog=data/decals/decals_master_catalog.csv`

`question=smooth`

`catalog_dir=data/decals/prepared_catalogs/decals_smooth_may`

`shard_dir=data/decals/shards/decals_smooth_may`

`experiment_dir=data/experiments/decals_smooth_may`

**Create master catalog** (could give command-line args)

`dvc run -o $master_catalog -f $master_catalog.dvc -d zoobot/active_learning/prepare_catalogs.py -d data/decals/disk_catalog.fits -d data/decals/classifications/streaming/classifications.csv python zoobot/active_learning/prepare_catalogs.py`

**Define experiment (create experiment catalogs)**

`dvc run -d $master_catalog -d zoobot/active_learning/define_experiment.py -o $catalog_dir -f $catalog_dir.dvc python zoobot/active_learning/define_experiment.py --master-catalog=$master_catalog --question=$question --save-dir=$catalog_dir`

**Create shards**

Real:

`dvc run -d $catalog_dir -d zoobot/active_learning/make_shards.py -o $shard_dir -f $shard_dir.dvc python zoobot/active_learning/make_shards.py --labelled-catalog=$catalog_dir/labelled_catalog.csv --unlabelled-catalog=$catalog_dir/unlabelled_catalog.csv --eval-size=5000 --shard-dir=$shard_dir --max-unlabelled=40000`

<!-- --max-labelled=99999  -->
Sim:

`dvc run -d $catalog_dir -d zoobot/active_learning/make_shards.py -o $shard_dir -f $shard_dir.dvc python zoobot/active_learning/make_shards.py --labelled-catalog=$catalog_dir/simulation_context/labelled_catalog.csv --unlabelled-catalog=$catalog_dir/simulation_context/unlabelled_catalog.csv --eval-size=2500 --shard-dir=$shard_dir`

**Run Simulation**

`dvc run -d $shard_dir -d $catalog_dir -d production/run_simulation.sh -o $experiment_dir -f $experiment_dir.dvc ./production/run_simulation.sh $catalog_dir $shard_dir $experiment_dir --test`

**Run Live**

Create instructions

`instructions_dir=$experiment_dir/instructions`

`dvc run -d $shard_dir -d $catalog_dir -d production/create_instructions.sh -o $instructions_dir -f $experiment_dir.dvc ./production/create_instructions.sh $catalog_dir $shard_dir $experiment_dir --panoptes`
<!-- add --test for test mode, --panoptes for real oracle/uploads -->

Run first iteration (manually setting folders)

`experiment_dir=data/experiments/decals_smooth_may`
`instructions_dir=$experiment_dir/instructions`

`previous_iteration_dir=""`

`this_iteration_dir=$experiment_dir/iteration_0`
<!-- 
`dvc run -d production/run_iteration.sh -d $instructions_dir -o $this_iteration_dir -f $this_iteration_dir.dvc ./production/run_iteration.sh  $experiment_dir $instructions_dir $previous_iteration $this_iteration "--test"` -->

`dvc run -d $shard_dir -d $instructions_dir -o $this_iteration_dir -f $this_iteration_dir.dvc python zoobot/active_learning/run_iteration.py --instructions-dir=$instructions_dir --this-iteration-dir=$this_iteration_dir --previous-iteration-dir=$previous_iteration_dir`

And going forwards:

`experiment_dir=data/experiments/decals_smooth_may`
`instructions_dir=$experiment_dir/instructions`
<!-- this does need to be defined! -->

`iteration_n=1`

`previous_iteration_dir=$experiment_dir/iteration_$(($iteration_n - 1))`
<!-- bash uses this syntax to evaluate in math context -->

`this_iteration_dir=$experiment_dir/iteration_$iteration_n`

`python zoobot/active_learning/run_iteration.py --instructions-dir=$instructions_dir --this-iteration-dir=$this_iteration_dir --previous-iteration-dir=$previous_iteration_dir`