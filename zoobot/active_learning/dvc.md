
# Training ML

Begin using GZD-C/DR5 final volunteer catalog. See zoobot/notebooks/catalogs/merge_catalogs.ipynb.

## ARC

Activate conda on ARC outside of a job:

    module load python/anaconda3/2019.03
    source activate /data/phys-zooniverse/chri5177/envs/zoobot

### Create Master Catalog

rsync -azv -e 'ssh -A -J chri5177@oscgate.arc.ox.ac.uk' /home/walml/repos/zoobot/current_final_dr5_result.parquet chri5177@arcus-htc:/data/phys-zooniverse/chri5177/repos/zoobot/data/decals

Classifications collected to-date, output by gz-panoptes-reduction as classifications.csv. Includes iauname column, to match with catalog.

`initial_classifications=data/decals/current_final_dr5_result.parquet`

`master_catalog=data/decals/decals_master_catalog.csv`

`dvc run -n master_catalog -o $master_catalog -d $initial_classifications python zoobot/science_logic/prepare_catalogs.py no_catalog $initial_classifications $master_catalog`


### Define experiment (create experiment catalogs)

`catalog_dir=data/decals/prepared_catalogs/decals_dr`

`dvc run -n experiment_catalog -d $master_catalog -d zoobot/science_logic/define_experiment.py -o $catalog_dir python zoobot/science_logic/define_experiment.py --master-catalog=$master_catalog --save-dir=$catalog_dir`

Optionally add --filter to select a subset of galaxies. 

Optionally set --sim-fraction to control fraction of galaxies to pretend are unlabelled in simulation_context catalogs. Does not affect real catalogs.

For active learning tests, you will use the sim catalogs. For real active learning, or `normal' offline training, you will use the real catalogs.

### Create shards

Real:

Use make_shards.sh

rsync -azv -e 'ssh -A -J chri5177@oscgate.arc.ox.ac.uk' /home/walml/repos/zoobot/arc/make_shards.sh chri5177@arcus-htc:/data/phys-zooniverse/chri5177/repos/zoobot/arc/make_shards.sh

sbatch arc/make_shards.sh

Sim:

Also use make_shards.sh, pointing to the simulation_context subfolder of the prepared catalogs.

### Run Offline (on labelled only)

Use offline_training.sh

rsync -azv -e 'ssh -A -J chri5177@oscgate.arc.ox.ac.uk' /home/walml/repos/zoobot/arc/offline_training.sh chri5177@arcus-htc:/data/phys-zooniverse/chri5177/repos/zoobot/arc/offline_training.sh

sbatch offline_training.sh

**Run Simulation**

    catalog_dir=data/gz2/prepared_catalogs/all_2p5_unfiltered
    shard_dir=data/gz2/shards/all_sim_2p5_unfiltered_300

    experiment_dir=data/experiments/live/latest
    instructions_dir=$experiment_dir/instructions

    export PYTHON=/home/walml/anaconda3/envs/zoobot/bin/python

    n_iterations=2

Args are baseline, test, and panoptes. Use ' ' to leave blank - the space is crucial.

    ./production/run_simulation.sh $n_iterations $catalog_dir $shard_dir $experiment_dir 'active_test'

    <!-- dvc run --ignore-build-cache -d $shard_dir -d $catalog_dir -d production/run_simulation.sh -o $experiment_dir -f $experiment_dir.dvc ./production/run_simulation.sh $n_iterations $catalog_dir $shard_dir $experiment_dir 'baseline' -->

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