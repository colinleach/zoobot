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

    active_config.prepare_run_folders()

    assert all([os.path.exists(subdir) for subdir in subdirs])
    assert os.path.exists(active_config.db_loc)


@pytest.mark.xfail()
def test_run(active_config_ready, tmpdir, monkeypatch, catalog_random_images, tfrecord_dir, acquisition_func):
    # TODO need to test we're using the estimators we expect, needs refactoring first
    # catalog_random_images is a required arg because the fits files must actually exist

    def mock_load_predictor(loc):
        # assumes run is configured for 3 iterations in total
        with open(os.path.join(loc, 'dummy_model.txt'), 'r') as f:
            training_records = json.load(f)
        if len(training_records) == 1:
            assert 'initial_train' in training_records[0]
            return 'initial train only'
        if len(training_records) == 2:
            return 'one acquired record'
        if len(training_records) == 3:
            return 'two acquired records'
        else:
            raise ValueError('More training records than expected!')
    monkeypatch.setattr(active_learning.make_predictions, 'load_predictor', mock_load_predictor)

    def mock_get_samples_of_subjects(model, subjects, n_samples):
        # only give the correct samples if you've trained on two acquired records
        # overrides conftest
        assert isinstance(subjects, list)
        example_subject = subjects[0]
        assert isinstance(example_subject, dict)
        assert 'matrix' in example_subject.keys()
        assert isinstance(n_samples, int)

        response = []
        for subject in subjects:
            if model == 'two acquired records':
                response.append([np.mean(subject['matrix'])] * n_samples)
            else:
                response.append(np.random.rand(n_samples))
        return np.array(response)
    monkeypatch.setattr(active_learning.make_predictions, 'get_samples_of_subjects', mock_get_samples_of_subjects)

    # TODO mock iterations as a whole, piecemeal moved to iterations
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
        acquired_shards = json.load(f)[1:]  # includes the initial shard, which is unsorted
    
    matrix_means = []
    for shard in acquired_shards:
        subjects = read_tfrecord.load_examples_from_tfrecord(
            [shard], 
            read_tfrecord.matrix_label_id_feature_spec(active_config_ready.shards.initial_size, active_config_ready.shards.channels)
        )
        shard_matrix_means = np.array([x['matrix'].mean() for x in subjects])

        # check that images have been saved to shards in monotonically decreasing order...
        assert all(shard_matrix_means[1:] < shard_matrix_means[:-1])
        matrix_means.append(shard_matrix_means)
    # ...but not across all shards, since we only predict on some shards at a time
    all_means = np.concatenate(matrix_means)
    assert not all(all_means[1:] < all_means[:-1])
