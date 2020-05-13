
## Locally

**Run new reduction (optional)**

*Check get_latest.py carefully first to make sure you're not deleting anything!*

`dvc run -o data/decals/classifications/classifications.csv python -f data/decals/classifications.csv.dvc ../gzreduction/main.py`

<!-- `dvc push -r s3 data/decals/classifications.dvc` -->

## EC2

<!-- `shard_dir=data/decals/shards/decals_weak_bars_sim` -->
<!-- `question=bar` -->
<!-- `catalog_dir=data/decals/prepared_catalogs/decals_weak_bars_launch`
`shard_dir=data/decals/shards/decals_weak_bars_launch`
`experiment_dir=data/experiments/simulation/decals_weak_bars_launch_test` -->

**Specify input data**

Catalog (i.e. big list) of all galaxies, including iauname, fits_loc, png_loc columns. No labels required or expected.

`initial_catalog=/media/walml/beta/decals/catalogs/decals_dr5_uploadable_master_catalog_nov_2019.csv`

Classifications collected to-date, output by gz-panoptes-reduction as classifications.csv. Includes iauname column, to match with catalog.

`initial_classifications=/media/walml/beta/decals/results/classifications_2019_11_27.csv`

**Create master catalog**

Will merge the catalog and classifications to-date here. Also remove duplicates and tweak filenames, if needed. Deprecated for GZ2, for now.

`master_catalog=data/decals/decals_master_catalog.csv`

`dvc run -o $master_catalog -f $master_catalog.dvc -d zoobot/science_logic/prepare_catalogs.py -d $initial_catalog -d $initial_classifications python zoobot/science_logic/prepare_catalogs.py $initial_catalog $initial_classifications $master_catalog`


**Define experiment (create experiment catalogs)**

`catalog_dir=data/decals/prepared_catalogs/decals_multiq`

`dvc run -d $master_catalog -d zoobot/science_logic/define_experiment.py -o $catalog_dir -f $catalog_dir.dvc python zoobot/science_logic/define_experiment.py --master-catalog=$master_catalog --save-dir=$catalog_dir`

**Create shards**

Real:

`shard_dir=data/decals/shards/decals_multiq_128`

`dvc run -d $catalog_dir -d zoobot/active_learning/make_shards.py -o $shard_dir -f $shard_dir.dvc python zoobot/active_learning/make_shards.py --labelled-catalog=$catalog_dir/labelled_catalog.csv --unlabelled-catalog=$catalog_dir/unlabelled_catalog.csv --eval-size=2500 --shard-dir=$shard_dir --max-unlabelled=40000 --img-size 128`

Sim:

`shard_dir=data/decals/shards/decals_multiq_128_sim_init_1800_featp5_facep5`

`dvc run -d $catalog_dir -d zoobot/active_learning/make_shards.py -o $shard_dir -f $shard_dir.dvc python zoobot/active_learning/make_shards.py --labelled-catalog=$catalog_dir/simulation_context/labelled_catalog.csv --unlabelled-catalog=$catalog_dir/simulation_context/unlabelled_catalog.csv --eval-size=1000 --shard-dir=$shard_dir --img-size 128 --max-labelled 5000`

**Run Simulation**

`export PYTHON=/home/walml/anaconda3/envs/zoobot/bin/python`
or
    export PYTHON=$DATA/envs/zoobot/bin/python

`experiment_dir=data/experiments/live/latest`

`instructions_dir=$experiment_dir/instructions`

`n_iterations=5`

`dvc run --ignore-build-cache -d $shard_dir -d $catalog_dir -d production/run_simulation.sh -o $experiment_dir -f $experiment_dir.dvc ./production/run_simulation.sh $n_iterations $catalog_dir $shard_dir $experiment_dir --baseline --test ''`

**Run Live**

Create instructions

`experiment_dir=data/experiments/decals_multiq_128`

`mkdir $experiment_dir`

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