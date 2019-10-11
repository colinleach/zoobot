# Service Design

## Aims

Run active learning daily on Galaxy Zoo.
Read this alongside the [service diagrams](https://www.lucidchart.com/documents/view/3af59d25-f169-4963-be7d-3218ef37f3aa).

## Local Preparation

### 1. Classifications

The bulk of previous classifications should already be reduced, so that each active learning iteration will only need to pull new classifications from the last full reduction.

Run [mwalmsley/gz-panoptes-reduction/get_latest.py](github.com/mwalmsley/gz-panoptes-reduction/get_latest.py) to execute a full reduction. See the instructions there.

Classifications should be placed under `zoobot/data/decals/classifications/classifications.csv`.

You need to run this periodically to prevent the gradual accumulation of classifications from slowing down each iteration. Make sure this is up-to-date before creating new master catalogs (below).

<!-- **Be careful to:**
- Do not wipe the working directory if it has major previous classifications -->

### 2. Master Catalogs

[zooniverse/decals](github.com/zooniverse/decals) or [mwalmsley/gz-panoptes-reduction/gz2](github.com/mwalmsley/gz-panoptes-reduction/gz2) provide catalogs of png image locations. 
`prepare_catalogs.py` tweaks these (for DECALS, joins with previous classifications) to create 'master catalogs' of all image locations and classifications to-date.
These are at `data/{survey}/{survey}_master_catalog.csv`.

## Ad Hoc Catalog Preparation

The master catalogs are generic. `define_experiment.py` takes a master catalog, defines the task to predict (i.e. creates the `label` and `id_str` columns), applies any ad-hoc changes, and splits into labelled (retired) and unlabelled (not retired) catalogs.

The labelled catalog is then further split into `mock_labelled.csv` and `mock_unlabelled.csv` for simulations, and `oracle.csv` is created to record the known-but-hidden labels for `mock_unlabelled.csv`.

These are at `data/{survey}/prepared_catalogs/{catalog name}`

## Shards

The labelled and unlabelled catalogs (whether 'real' or mock) are 
- split (labelled only)
- converted to tfrecords
- indexed in an sqlite database

The ShardConfig object tracks this process, and is written to disk as json so that later scripts know where the new shards are and how they were created.

## Execution

By now, you should have:
- labelled and unlabelled catalogs of images
- shards created using those catalogs

Run active learning using these two scripts:
- `create_instructions.py` to create `instructions` from command line args and sensible hard-coded values
- `run_iteration.py`to run an iteration, with `instructions` and an `{iteration_name}` as input and a new `{iteration_name}` as output

Each script takes immutable folders as input, and produce immutable folders as output.

You will need an EC2 instance for GPU power. See 'EC2 Configuration'.

For simulations, run `run_simulation.sh` on an EC2 instance. This:
- Calls `create_instructions.py` once
- Loops over `run_iteration.py`

For production, we manually run `create_instructions.py` `run_iteration.py` to get started. 
Then, on a schedule:
- Spin up a new EC2 instance (preserving the file system from last run)
- Run `run_iteration.py` with the previous (latest?) `{iteration_name}` as an input directory.
- Spin down

## EC2 Configuration

Setting up the EC2 instance is a manual, one-step process. 
The key commands are in `ec2_environment.sh`

The AMI defines the software configuration for the instance. 
Starting from the AWS DL AMI, requirements are installed from each repo.
These are saved under the root disk (75GB), and fixed in a snapshot.

To load the data and previous state, mount the volume `working_volume`. 
See `ec2_environment.sh` for more.

I may schedule spinning up and running an iteration with a shell script.

## EC2 Instance File Structure

For now, keep it simple: all state is preserved on a single EBS volume called `working_volume`.

- root (EBS volume mounted here)
    - auth
        - `zooniverse_login.json`
        - `github`
        - `github.pub`
    - repos
        - gz-panoptes-reduction
            - data (use GZ2 reduction in here for GZ2 master catalog)
        - zoobot
            - data
                - gz2
                    - `gz2_classifications_and_subjects.csv`
                    - png
                    - shards
                - decals
                    - `joint_catalog_selected_cols.csv`
                    - `decals_master_catalog.csv`
                    - png_native
                    - shards
                        - {shard name}
                - experiments
                    - simulations
                        - {experiment name}
                    - live
                        - {experiment name}
        - shared-astro-utils

Under each {experiment name}:
- instructions (i.e. `instructions_dir`)
- iteration_{n}


## Getting and Requesting Labels

I need to update mock_panoptes to give raw classification output, exactly like the API. `iterations.py` will then run the reduction pipeline on raw classifications to get labels.

Requesting labels will pull all raw classifications after some last classification_id (defined by command line and recorded in `instructions`)matching the last id used to create the original `{survey}_reduced_classifications.csv`. A reduction will then be run on only those raw classifications, for every iteration. This is messy, but is easy and will work okay for now.

For the moment, let's get the stateless system working first and *then* change the mocking to be more complicated.


---

 
### State Before Iterations

Shards record the file paths on creation to `shard_config` object and then to disk via json. The (reloaded) `shard_config` file paths need to be updated later if the shards are subsequently moved.

### State Between Iterations

The basic problem is: I use this level to share state just before and between iterations. In production, I need to write all this state to disk.
Instructions for how (and when) to execute iterations.

**Active Learning Initial Setup**

Inputs: detailed instructions, shards to use
Outputs: 

- Working directory (`run_dir`), n_iterations, subjects_per_iter, shards_per_iter
- final_size
- Wipe clean any existing data in `run_dir` working directory (*Iterations should no longer take place within `run_dir`, this should now refer to immutable instructions folder*)
- Loads `shard_config` from json and updates in-memory to have correct file paths for `shard_dir`, which may have changed since `shard_config` was created
- Copies catalog database from `shard_config`, to be further copied and modified by each iteration
- TrainCallableParams instance: initial size, final size, warm start, eval_tfrecord_loc, test (bool)

*Save all of this as immutable state*

**Based on what we've done so far, what args should we provide to the next iteration?**

Inputs: 

train_callable and acquisition_func are stateless. 
They can be reconstructed with `get_train_callable` and `get_acquisition_func` before running each iteration.

### State Within Iteration

**Iterations must be isolated (no side effects outside `iteration_dir`) and (once run) immutable.**


With defined instructions, execute an iteration.
- Working folder
- Name
- Prediction shards
- Database of id_str, label, total_votes and file_loc for galaxies in prediction shards. Partially labelled.
- Evaluation shards (for metrics)
- Train callable (which currently includes `params` state via closure)

