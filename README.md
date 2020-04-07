



[![Build Status](https://travis-ci.org/RustyPanda/zoobot.svg?branch=master)](https://travis-ci.org/RustyPanda/zoobot)
[![Maintainability](https://api.codeclimate.com/v1/badges/dcd9b142609237a90574/maintainability)](https://codeclimate.com/github/RustyPanda/zoobot/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/dcd9b142609237a90574/test_coverage)](https://codeclimate.com/github/RustyPanda/zoobot/test_coverage)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

# Zoobot

**This repository contains the models and data needed to run active learning on Galaxy Zoo.**

Scripts in this directory apply the Zoobot package (below).

- `run_zoobot_on_panoptes.py` trains and evaluates a model on a train/test split of Panoptes galaxies. This is useful for developing models. `status.md` records each experiment (manually).
- `panoptes_to_tfrecord.py` and `gz2_to_tfrecord.py` convert the respective download catalogs into tfrecords suitable for models.

I feel like there was a script to run models on GZ2, but I'm not sure where it's gone. Perhaps I should generalise `run_zoobot_on_panoptes.py`.

## Zoobot

This is the Python package. **All scripts should be run from this directory.**

## Installation

### Local development

    git clone git@github.com:mwalmsley/zoobot.git
    pip install -r zoobot/requirements.txt
    pip install -e zoobot

### Docker

    The Dockerfile in the repo root works, but isn't built and saved anywhere public. I probably shouldn't bother until public release.

### Data Required

The Galaxy Zoo 2 and Nasa Sloan Atlas galaxy catalogs required are available externally from [here](data.galaxyzoo.org) and [here](https://www.sdss.org/dr13/manga/manga-target-selection/nsa/).

## Usage

We provide example scripts for using the `zoobot` package in the root directory. 

- `offline_training.py` will create and train a model on existing (i.e. offline) data.
- `panoptes_to_tfrecord.py` and `gz2_to_tfrecord.py` convert the respective download catalogs into tfrecords suitable for models.
- `run_zoobot_on_panoptes.py` used to train and evaluates a model on a train/test split of either GZ2 or GZ DECALS galaxies. It has been deprecated in favor of `offline_training.py`.

## How does it work?

Active learning at scale is fiddly because:
- everything needs to be saved to disk, and reloaded at the next iteration
- There's a lot to abstract away: the model, the acquisition function, and the active learning setup itself (e.g. subjects per batch).

We handle both problem at once with three classes:`TrainCallableFactory`, `AcquisitionCallableFactory`, and `Instructions`.

`TrainCallableFactory` handles defining the model. 

Your model should be defined by a set of fixed arguments (e.g. number of layers in the model) which don't change per iteration. `TrainCallableFactory.get()` will create and return your model from those fixed arguments. The fixed arguments are saved to disk, allowing the model to be recreated later. 

Actually training the model often requires some iteration-specific arguments (e.g. learning rate). The model function returned by `TrainCallableFactory.get()` expects only those iteration-specific arguments. `iteration.py` will handle passing it those arguments.

`AcquistionCallableFactory` does exactly the same for your acquisition function, which will have some fixed arguments and (potentially) some per-iteration arguments. Per-iteration arguments aren't implemented yet.

`Instructions` handles your instructions for actually running active learning - how many subjects to request labels for per iteration, where all the data lives, etc. This is pretty much just a fancy dictionary that saves and reloads from disk as needed. It has a `use_test_mode()` method which overrides the specified instructions to do very short iterations, which is handly for debugging.

## Alright, how do I run it?

### Data Prep

You should have a catalog of all your subjects you know about them. We call this the `master catalog`. Many subjects will have missing data that you'd like to learn to predict. The master catalog stays fixed until you've finished your active learning process.

From the master catalog, we pick some data to learn to predict. In practice, we move the data we're interested in predicting to specific columns that `zoobot` expects. We call this new catalog an `experiment catalog`.  We also create a new `simulation catalog` where some of the labels we actually know has been hidden. Why? Before production, it's a good idea to run a Simulation, hiding some labels that we actually know and revealing them (by reading the master catalog) as the model asks. 

    python zoobot/active_learning/define_experiment.py --master-catalog=$master_catalog --question=$question --save-dir=$catalog_dir

Lastly, we serialize the data to tfrecord shards so that tensorflow can read it efficiently and shuffle it out-of-memory. from either the full Experiment catalog, or the partially-hidden Simulation catalog.

Set $catalog_dir as the directory with your Experiment, or the `simulation_context` subdirectory with your Simulation catalog. Set $shard_dir as the output directory for the shards. Then you can make tfrecord shards with:

    python zoobot/active_learning/make_shards.py --labelled-catalog=$catalog_dir/labelled_catalog.csv --unlabelled-catalog=$catalog_dir/unlabelled_catalog.csv --eval-size=5000 --shard-dir=$shard_dir --max-unlabelled=40000

### Running


**Run Simulation**

    ./production/run_simulation.sh $catalog_dir $shard_dir $experiment_dir

    Add --test to debug with quick iterations.


**Run Live**

Create instructions:

    instructions_dir=$experiment_dir/instructions
    
    ./production/create_instructions.sh $catalog_dir $shard_dir $experiment_dir 

Add --test for quick iterations, --panoptes for real oracle/uploads to Panoptes, the Zooniverse backend server.

Run the first iteration:

    experiment_dir=data/experiments/decals_smooth_may
    instructions_dir=$experiment_dir/instructions

    previous_iteration_dir=""
    this_iteration_dir=$experiment_dir/iteration_0

    python zoobot/active_learning/run_iteration.py --instructions-dir=$instructions_dir --this-iteration-dir=$this_iteration_dir --previous-iteration-dir=$previous_iteration_dir

And going forwards:

    experiment_dir=data/experiments/decals_smooth_may
    instructions_dir=$experiment_dir/instructions

    iteration_n=1
    this_iteration_dir=$experiment_dir/iteration_$iteration_n
    previous_iteration_dir=$experiment_dir/iteration_$(($iteration_n - 1))

*The $(( expression )) syntax tells bash to evaluate the math expression within.*
    
    python zoobot/active_learning/run_iteration.py --instructions-dir=$instructions_dir --this-iteration-dir=$this_iteration_dir --previous-iteration-dir=$previous_iteration_dir

I've made an attempt at a shell script for this, but it's not ready yet. I'll probably switch to airflow.


## Zoobot Folders
- `estimators` includes custom TensorFlow models (and associated training routines and input utilities) for galaxy classification. 
- `active_learning` applies those models and routines in an active learning loop. Currently, previous classifications are used as an oracle. **See the Readme in this folder for more details.**
- `get_catalogs` is used to download GZ2 classifications. Panoptes classifications have been refactored to the repo `gz-panoptes-reduction`.
- `tests` contains unit tests for the package. Look here for examples!
- `tfrecord` has various useful utilities for reading and writing galaxy catalogs to tfrecords.
- `uncertainty` has code to find the coverage fractions of trained models


## Contact Us

This code was written by [Mike Walmsley](walmsley.dev). Please get in touch by [email](mailto:mike.walmsley@physics.ox.ac.uk).

### Legacy Code
- `embeddings` is ancient code to visualse galaxies in tensorboard. *Not currently used.*
- `examples` is a few examples I found helpful when designing the models. *Not currently used.*
- `illustris` is ancient code used to test model performance at identifying mergers in Illustris. Review this before starting a merger project. *Not currently used*
- `prn` is hackathon code used to write Planetary Response Network images to tfrecord. **This should definitely be extracted**.

## Data

`basic_split` holds tfrecords used by `zoobot/estimators` to evaluate models

The remaining folders are used for active learning. See the `zoobot/active_learning` readme for more details.


## Results

This folder currently contains:
- Models trained on a basic split, to be compared for performance and iterated
- Metrics from active learning cycles

This is a bit of a mess and I should ponder how to do this cleanly.


## Analysis

- `bayesian_cnn` (which should be renamed) has some early results using dropout, including timings
- `uncertainty` uses `check_uncertainty.py` and `zoobot/uncertainty` to investigate how trained models perform and select new subjects


## Legal Stuff

Copyright (C) 2019 - Mike Walmsley

This program is NOT free software and you MAY NOT use it in any way without express permission of the license holder(s).