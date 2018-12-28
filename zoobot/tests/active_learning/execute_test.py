import os
import json
import time
import copy

import pytest
import numpy as np
from zoobot.tests.active_learning import conftest

from zoobot.tfrecord import read_tfrecord
from zoobot.active_learning import active_learning, make_shards, execute



def test_prepare_run_folders(active_config):
    assert os.path.isdir(active_config.run_dir)  # permanent directory for dvc control
    subdirs = [
        active_config.requested_fits_dir, 
        active_config.requested_tfrecords_dir
    ]
    assert not any([os.path.exists(subdir) for subdir in subdirs])
    assert not os.path.exists(active_config.db_loc)
    assert not os.path.exists(active_config.train_records_index_loc)

    active_config.prepare_run_folders()

    assert all([os.path.exists(subdir) for subdir in subdirs])
    assert os.path.exists(active_config.db_loc)
    assert os.path.exists(active_config.train_records_index_loc)



def test_run(active_config_ready, tmpdir, monkeypatch, catalog_random_images, tfrecord_dir, acquisition_func):
    # TODO need to test we're using the estimators we expect, needs refactoring first
    # catalog_random_images is a required arg because the fits files must actually exist

    def train_callable(estimators_dir, train_tfrecord_locs):
        # pretend to save a model in subdirectory of estimator_dir
        assert os.path.isdir(estimators_dir)
        subdir_loc = os.path.join(estimators_dir, str(time.time()))
        os.mkdir(subdir_loc)

    def mock_load_predictor(loc):
        return None
    monkeypatch.setattr(active_learning.make_predictions, 'load_predictor', mock_load_predictor)

    monkeypatch.setattr(active_learning.make_predictions, 'get_samples_of_subjects', conftest.mock_get_samples_of_subjects)

    def mock_get_labels(subject_ids):  
        # don't actually read from saved catalog, just make up
        # TODO check that these are correctly saved to db and preserved over iterations
        return [np.random.rand() for n in range(len(subject_ids))]
    monkeypatch.setattr(active_learning.mock_panoptes, 'get_labels', mock_get_labels)

    active_config_ready.run(train_callable, acquisition_func)

    # verify the folders appear as expected
    for iteration_n in range(active_config_ready.iterations):
        # copied to iterations_test.py
        # separate dir for each iteration
        iteration_dir = os.path.join(active_config_ready.run_dir, 'iteration_{}'.format(iteration_n))
        assert os.path.isdir(iteration_dir)
        # which has a subdir recording the estimators
        estimators_dir = os.path.join(iteration_dir, 'estimators')
        assert os.path.isdir(estimators_dir)
        # if initial estimator was provided, it should have been copied into the of 0th iteration subdir
        if iteration_n == 0:
            if active_config_ready.initial_estimator_ckpt is not None:
                assert os.path.isdir(os.path.join(estimators_dir, active_config_ready.initial_estimator_ckpt))
        else:
            if active_config_ready.warm_start:
                # should have copied the latest estimator from the previous iteration
                latest_previous_estimators_dir = os.path.join(active_config_ready.run_dir, 'iteration_{}'.format(iteration_n - 1), 'estimators')
                latest_previous_estimator = active_learning.get_latest_checkpoint_dir(latest_previous_estimators_dir)  # TODO double-check this func!
                assert os.path.isdir(os.path.join(estimators_dir, os.path.split(latest_previous_estimator)[-1]))
    
    # read back the training tfrecords and verify they are sorted by order of mean
    with open(active_config_ready.train_records_index_loc, 'r') as f:
        training_shards = json.load(f)[1:]  # includes the initial shard, which is unsorted
    
    for shard in training_shards:
        subjects = read_tfrecord.load_examples_from_tfrecord(
            [shard], 
            read_tfrecord.matrix_label_id_feature_spec(active_config_ready.shards.initial_size, active_config_ready.shards.channels)
        )
        matrix_means = np.array([x['matrix'].mean() for x in subjects])

        # check that images have been sorted into monotonically decreasing order, even across shards
        assert all(matrix_means[1:] < matrix_means[:-1])
