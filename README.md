



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

Clone the repo

    git clone git@github.com:mwalmsley/zoobot.git
    pip install -r zoobot/requirements.txt
    pip install -e zoobot

## Usage

We provide example scripts for using the `zoobot` package in the root directory. 

- `run_zoobot_on_panoptes.py` trains and evaluates a model on a train/test split of either GZ2 or GZ DECALS galaxies. This is useful for developing models. 
- `panoptes_to_tfrecord.py` and `gz2_to_tfrecord.py` convert the respective download catalogs into tfrecords suitable for models.

To simulate active learning, see the readme in `zoobot/active_learning`.

*Note that the Galaxy Zoo 2 and Nasa Sloan Atlas galaxy catalogs required are available externally from [here](data.galaxyzoo.org) and [here](https://www.sdss.org/dr13/manga/manga-target-selection/nsa/).*

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