# Active Learning for Panoptes

This document details how to run active learning on an existing catalog. This is a two-step process.
1. Write images without labels into many tfrecord chunks, called shards, plus an additional labelled tfrecord.
2. Run a model to learn from the labelled tfrecord, and then (iterating through each shard) pick images for labelling.

## DVC Pipeline

If you don't have a new catalog or new images, skip ahead with `dvc pull make_shards.dvc`. 
This will download the 
Otherwise, you can run `dvc repro make_shards.dvc` **locally** to. `dvc repro` will scan for file changes since the pipeline was last run 
Below

`catalog_loc=data/panoptes_predictions_selected.csv`
`fits_dir=data/fits_native`
`shard_dir=data/shards/si128_sf64_ss4096`
`run_dir=data/runs/example_run`

### Data from Outside Repo

We need a catalog and images. 

`latest_raw_catalog_loc` points to the latest Panoptes reduction export (including the NSA catalog details).
Copy this locally, so we have the full record of labelled galaxies and their (original) locations on disk.


`latest_raw_catalog_loc=data/2018-11-05_panoptes_predictions_with_catalog.csv`
`dvc run -o $latest_raw_catalog_loc -f get_raw_catalog.dvc cp /data/galaxy_zoo/decals/panoptes/reduction/output/2018-11-05_panoptes_predictions_with_catalog.csv $latest_raw_catalog_loc`

The native size images are 250GB (!), so (for now, running a historical simulation) we prefer not to copy all of them.
`create_panoptes_only_files.py` will pick out the images which already have labels, copy them to this repo, and write a new catalog `panoptes_predictions_selected.csv` with the updated fits locations.


`dvc run -d $latest_raw_catalog_loc -O $fits_dir -o $catalog_loc -f get_fits.dvc python zoobot/active_learning/create_panoptes_only_files.py --new_fits_dir=$fits_dir --old_catalog_loc=$latest_raw_catalog_loc --new_catalog_loc=$catalog_loc`

From this point, we only care about files in the repo.

### Shards

Now we have the data to create shards. 

`update_catalog_fits_loc.py` will adjust the catalog `fits_loc` column to correctly point to the ec2 location.



`make_shards.py` will 
- Take the first 1024 images as training and test data (random 80/20), and save galaxy ids and labels for the remainder in `zoobot/active_learning/oracle.csv` to be used by `mock_panoptes.py` to simulate an oracle.
- Pretending that the remaining images are unlabelled, write each image to a shard and create a database recording where each image is. This database will also store the revealed labels and latest acquisition values, to be filled in later.
- Record the shard and database locations, and other metadata, in a 'shard config' (json-serialized dict). This lets us use these shards later.


`dvc run -d $catalog_loc -d data/fits_native -o zoobot/active_learning/oracle.csv -o $shard_dir -f make_shards.dvc python zoobot/active_learning/make_shards.py --catalog_loc=$catalog_loc --shard_dir=$shard_dir`

### Execution

**Already run the commands above?** `dvc pull make_shards.dvc` **will skip to here. Helpful!**

Finally, we can run the actual active learning loop. Thanks to the shard config, we can read and re-use the shards without having to recreate them each time.
`dvc run -d $shard_dir -d zoobot/active_learning/oracle.csv -o $run_dir -f execute_al.dvc python zoobot/active_learning/execute.py --shard_config=$shard_dir/shard_config.json --run_dir=$run_dir`
OR
`dvc run -d $shard_dir -d zoobot/active_learning/oracle.csv -d zoobot -o $run_dir -f execute_al_baseline.dvc python zoobot/active_learning/execute.py --shard_config=$shard_dir/shard_config.json --run_dir=$run_dir --baseline=True`

shard_config is the config object describing the shards. run_dir is the directory to create run data (estimator, new tfrecords, etc).
Optionally, add --baseline=True to select samples for labelling randomly.
**Check that logs are being recorded**. They should be in the directory the script was run from (i.e. root).

## Optional: Run Tensorboard to Monitor

On local machine, open an SSH tunnel to forward the ports using the `-L` flag:
`ssh -i $key -L 6006:127.0.0.1:6006 $user@$public_dns`

Then, via that SSH connection (or another), run
`source activate tensorflow_p36`
`tensorboard --logdir=.`
to run a Tensorboard server showing both baseline and real runs, if available


## Optional: Save results to S3
`aws s3 sync $root/$shard_dir s3://galaxy-zoo/active-learning/runs/$run_dir`


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

