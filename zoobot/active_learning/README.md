### Active Learning for Panoptes

run_active_learning.py is the launch file. It configures everything.
Part of that configuration is done via estimator_from_disk.py

estimator_from_disk.py has all the configuration required for training and evaluating an estimator. run_active_learning.py creates the RunConfig object by passing extra args to estimator_from_disk.py

*I should have the sql database and training tfrecords on a network volume mount for docker, and active learning running on docker*

*for now, tests should pass on docker*

*then, tests should pass with data directories on network mount*

Directory Structure
- repo
    - tests
        - test_example_dir
- data (attached volume)
    - 

AWS S3 (partially copied locally on laptop)
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

S3 is designed as a data lake. S3 to EC2 transfers are around 10MB/s. S3 should **not** be used as a workspace for EC2 instances (and by extension, ECS containers)

### AWS EFS

Mirrors an active run within the S3 `active_learning_runs` folder. Also mirrors the unlabelled shards subfolder required for new images.
Mirror following [this link](https://n2ws.com/blog/how-to-guides/how-to-copy-data-from-s3-to-ebs).

EBS blocks are attached under /dev/sd{a, b...}

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

