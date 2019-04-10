# Service Design

## Aims

Run active learning daily on Galaxy Zoo.

## State

For now, let's assume that everything is happening on the same file system. This is the simplest thing to implement.

Each service block will take immutable folders as input, and produce immutable folders as output.

There are three types of folder expected to change often:
- `{shards_name}` for shards to load, and an oracle for simulations (`mock_panoptes.py` should give raw classifications, not reduced)
- `instructions` to define how each iteration should run (no longer in-memory)
- `{iteration_name}` containing the results of each iteration (including new label reduction subfolder)

We also need a few static folders and files:
- `{survey_png_dir}` with the image files
- `{survey}_reduced_classifications.csv` with reduced classifications *prior to any active learning*, for making shards

We can then run active learning with these stateless files:
- `create_instructions.py` to create `instructions` from command line args and sensible hard-coded values
- `run_iteration.py`to run an iteration, with `instructions` and an `{iteration_name}` as input and a new `{iteration_name}` as output

For production, we schedule:
- Spin up an EC2 instance with our filesystem and environment
- Run `run_iteration.py` with the previous (latest?) `{iteration_name}` as an input directory.
- Save the new filesystem to S3

I need to ask Adam how to best achieve this. 
This will require some state to track what's going on.

For simulations, we can do exactly the same except on a single filesystem and with trivial scheduling.

## EC2 Instance File Structure

Keep it simple: use a single EBS volume.

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
                    - png
                    - shards
                - results
                    - simulations
                        - {run name}
                    - live
                        - {run name}
        - shared-astro-utils


    Under each {run name}:
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

