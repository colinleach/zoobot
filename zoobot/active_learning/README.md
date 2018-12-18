# Active Learning for Panoptes

This document details how to run active learning on an existing catalog. This is a two-step process.
1. Write images without labels into many tfrecord chunks, called shards, plus an additional labelled tfrecord.
2. Run a model to learn from the labelled tfrecord, and then (iterating through each shard) pick images for labelling.

## DVC Pipeline

If you don't have a new catalog or new images, skip ahead with `dvc pull make_shards.dvc`. 
This will download the 
Otherwise, you can run `dvc repro make_shards.dvc` **locally** to. `dvc repro` will scan for file changes since the pipeline was last run 
Below


`external_catalog_loc=/data/galaxy_zoo/decals/panoptes/reduction/output/2018-11-05_panoptes_predictions_with_catalog.csv`
`external_fits_dir=/Volumes/alpha/decals/fits_native` (hardcoded into create_panoptes_only_files.py, only used to track dependencies)

`catalog_loc=data/panoptes_predictions_selected.csv`
`fits_dir=data/fits_native`
`shard_dir=data/shards/si128_sf64_ss4096`

### Data from Outside Repo

We need a catalog and images. 

`external_catalog_loc` points to the latest Panoptes reduction export (including the NSA catalog details).
The native size images are 250GB (!), so (for now, running a historical simulation) we prefer not to copy all of them.
`create_panoptes_only_files.py` will pick out the images which already have labels, copy them to this repo, and write a new catalog `panoptes_predictions_selected.csv` with the updated fits locations.

`dvc run -d $external_catalog_loc -d $external_fits_dir -O $fits_dir -o $catalog_loc -f get_fits.dvc python zoobot/active_learning/create_panoptes_only_files.py --old_fits_dir=$external_fits_dir --new_fits_dir=$fits_dir --old_catalog_loc=$external_catalog_loc --new_catalog_loc=$catalog_loc`

The -O option prevents fits_dir from being added to the cache. This is because dvc can't handle this many files. We can still call it as a -d later. Instead, sync to AWS manually:
`dvc run -d $fits_dir -f fits_to_s3.dvc aws s3 sync --size-only data/fits_native s3://galaxy-zoo/decals/fits_native`

From this point, we only care about files in the repo, and we can proceed on AWS EC2.

### Shards

Now we have the data to create shards. 

`make_shards.py` will 
- Take the first 1024 images as training and test data (random 80/20), and save galaxy ids and labels for the remainder in `zoobot/active_learning/oracle.csv` to be used by `mock_panoptes.py` to simulate an oracle.
- Pretending that the remaining images are unlabelled, write each image to a shard and create a database recording where each image is. This database will also store the revealed labels and latest acquisition values, to be filled in later.
- Record the shard and database locations, and other metadata, in a 'shard config' (json-serialized dict). This lets us use these shards later.

`dvc run -d $catalog_loc -d $fits_dir -d zoobot/active_learning/make_shards.py -o zoobot/active_learning/oracle.csv -o $shard_dir -f make_shards.dvc python zoobot/active_learning/make_shards.py --catalog_loc=$catalog_loc --shard_dir=$shard_dir`

### Execution

**Already run the commands above?** `dvc pull make_shards.dvc` **will skip to here. Helpful!**


Finally, we can run the actual active learning loop. Thanks to the shard config, we can read and re-use the shards without having to recreate them each time.

If you still need to acquire the data:
`dvc pull -r s3 make_shards.dvc`
`aws sync s3://galaxy-zoo/decals/fits_native data/fits_native` (takes a few minutes)
(eventually, can do dvc pull, but need to branch away from basic_split)

`execute.py` will:
- Create (or wipe) the run directory `run_dir`
- Run a training callable: currently, a model defined in `default_estimator_params.py` and tweaked in `execute.py`, with checkpoints recorded under the `estimator` directory. **Warning: wipes the checkpoints after each iteration!** All labelled subjects (training + previously acquired) are used.
- Run an acquisition callable: for each subject in the shards, record an acquisition value. Currently, this finds the mutual information of the model predictions. The --baseline=True flag replaces the mutual information with a random number generator, leading to random selection of subjects. 
- Use the shard db `catalog` table to look up the `fits_loc` of each high-acquisition subject, and save to a new tfrecord `requested_tfrecords/acquired_shard_{iteration_n}.tfrecord`
- Repeat, including the newly-acquired shard in the training pool
- Stop after a specified number of iterations, moving the log into the run directory

`run_dir=data/runs/al_mutual`
`dvc run -d $shard_dir -d zoobot/active_learning/oracle.csv -d $fits_dir -d zoobot -o $run_dir python zoobot/active_learning/execute.py --shard_config=$shard_dir/shard_config.json --run_dir=$run_dir`
OR baseline:
`baseline_dir=data/runs/al_baseline`
`dvc run -d $shard_dir -d zoobot/active_learning/oracle.csv -d $fits_dir -d zoobot -o $baseline_dir python zoobot/active_learning/execute.py --shard_config=$shard_dir/shard_config.json --run_dir=$baseline_dir --baseline=True`

shard_config is the config object describing the shards. run_dir is the directory to create run data (estimator, new tfrecords, etc).
Optionally, add --baseline=True to select samples for labelling randomly.
**Check that logs are being recorded**. They should be in the directory the script was run from (i.e. root).

Finally, upload the results.

`git add $run_dir.dvc` or `git add $baseline_dir.dvc`
`dvc push $run_dir.dvc -r s3`

## Generate Metrics

This will run locally or on EC2, but requires both `shard_dir` and `run_dir` to be pulled.
`dvc pull get_shards.dvc`
`dvc pull $run_dir.dvc`
`dvc pull $baseline_dir.dvc`

`output_dir=results/latest_metrics`
`dvc run -d $shard_dir -d $run_dir -d $baseline_dir -o $output_dir -f al_metrics.dvc python zoobot/active_learning/analysis.py --active_dir=$run_dir --baseline_dir=$baseline_dir --initial=512 --per_iter=256 --output_dir=$output_dir`

## Optional: Run Tensorboard to Monitor

On local machine, open an SSH tunnel to forward the ports using the `-L` flag:
`ssh -i $key -L 6006:127.0.0.1:6006 $user@$public_dns`

Then, via that SSH connection (or another), run
`source activate tensorflow_p36`
`tensorboard --logdir=.`
to run a Tensorboard server showing both baseline and real runs, if available

Directory Structure
- repo
    - tests
        - test_example_dir
- data (attached volume)
    - 

AWS S3
- galaxy_zoo (bucket, not folder)
    - decals
        - fits_native
        - png_native
        - etc, following alpha/decals
        - tfrecords
            - (current train/test records)
            - unlabelled_shards
                - decals_si_sf_...
                    - shard_0.tfrecord
                    - shard_1.tfrecord
        - active_learning_runs
            - (run name)
                - run_name.log
                - estimator
                    - model.ckpt-128
                - labelled_shards


**Setup**
Makes the shards and the database, which are both then copied by snapshot

**Run Data `run`** (fresh, save snapshot if needed)
- (run name)
    - run_db.db (for now)
    - labelled_shard_index.json
    - run_name.log
    - estimator
    - model.ckpt-128
    - labelled_shards

**Static Data `static_data`** (from immutable snapshot, perhaps directly from S3)
(from decals_si_sf_...)
    - shard_0.tfrecord
    - shard_1.tfrecord
    - static_shard_db.db



## General AWS Storage Notes

EFS can be shared between EC2 instances, and therefore between containers. Completed runs can be copied back to S3 after the job.

"You can use Amazon EFS file systems with Amazon ECS to export file system data across your fleet of container instances. That way, your tasks have access to the same persistent storage, no matter the instance on which they land."
https://docs.aws.amazon.com/AmazonECS/latest/developerguide/using_efs.html


S3 is data lake for everything needed for ML - primarily, the native fits files and common tfrecords
/Volumes/alpha is my pretend s3
EBS blocks are used to share this data locally
Static data snapshot before starting a run
With EC2, attach snapshot to instance (static_dir) and attach extra EBS to hold run data (run_dir)
This is a two-step process! estimator_dir and db should be made AFTER the snapshot

I need to be a bit more specific about exactly which variables change the snapshot, and therefore exactly what parts of active config should go where. I should also consider a snapshotconfig and a activerunconfig?

