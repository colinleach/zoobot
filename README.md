



[![Build Status](https://travis-ci.org/RustyPanda/zoobot.svg?branch=master)](https://travis-ci.org/RustyPanda/zoobot)
[![Maintainability](https://api.codeclimate.com/v1/badges/dcd9b142609237a90574/maintainability)](https://codeclimate.com/github/RustyPanda/zoobot/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/dcd9b142609237a90574/test_coverage)](https://codeclimate.com/github/RustyPanda/zoobot/test_coverage)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

# ZooBot

get_galaxy_zoo_catalog.py matches the published classifications table with Sandor's subject list (with AWS urls)
The matched file is saved as all_labels.csv
Currently, only spiral-related columns are saved

download_from_aws.py downloads each png from AWS
The catalog is updated with png_loc and png_ready, and saved as all_labels_downloaded.csv

gz2_to_tfrecord.py takes all_labels_downloaded.csv and
- does 80% random split into train/test 
- saves every image to tfrecord, along with label defined as int(subject['t04_spiral_a08_spiral_weighted_fraction'])
- this is a problem because some examples may have few spiral votes: a vote count cut is required

The tfrecord files are then used to train/test CNN.
spiral_experiment tells train_spiral_model what to do with Estimators
estimator_models defines the possible structures of each model
train_spiral_model creates the estimator and executes train/testing

input_utils (input_test) helps with parsing and queueing the tfrecords
stratify_images (stratify_images_test) checks that stratify works properly. Now outdated - default stratify used.
