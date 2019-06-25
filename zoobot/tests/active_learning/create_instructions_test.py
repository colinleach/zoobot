import pytest

import os
import json
import time
import copy

import numpy as np

from zoobot.tests.active_learning import conftest
from zoobot.estimators import run_estimator
from zoobot.tfrecord import read_tfrecord
from zoobot.active_learning import make_shards, iterations, create_instructions


@pytest.fixture()
def instructions_dir(tmpdir):
    return tmpdir.mkdir('instructions_dir').strpath

# TODO move to conftest?
@pytest.fixture(params=[{'warm_start': True}, {'warm_start': False}])
def train_callable_factory(request):
    return create_instructions.TrainCallableFactory(
        initial_size=128,
        final_size=64,
        warm_start=request.param['warm_start'],
        test=False
    )


@pytest.fixture()
def acquisition_callable_factory(request, baseline):
    return create_instructions.AcquisitionCallableFactory(
        baseline=baseline,
        expected_votes=10
    )

# TrainCallable tests
def test_train_callable_get(mocker, train_callable_factory):
    """Functional test for the train callable from train_callable_factory"""
    train_callable = train_callable_factory.get()
    assert callable(train_callable)

    # functional test investigating if train_callable really works, with run_estimator mocked
    from zoobot.estimators import run_estimator
    mocker.patch('zoobot.estimators.run_estimator.run_estimator')
    log_dir = 'log_dir'
    train_records = 'train_records'
    eval_records = 'eval_records'
    train_callable(log_dir, train_records, eval_records, learning_rate=0.001, epochs=2)
    run_estimator.run_estimator.assert_called_once()
    run_config = run_estimator.run_estimator.mock_calls[0][1][0]  # first call, positional args, first arg
    assert run_config.log_dir == log_dir
    assert run_config.train_config.tfrecord_loc == train_records
    assert run_config.eval_config.tfrecord_loc == eval_records
    assert run_config.warm_start == train_callable_factory.warm_start


def test_train_callable_save_load(train_callable_factory, instructions_dir):
    train_callable_factory.save(instructions_dir)
    loaded = create_instructions.load_train_callable(instructions_dir)
    assert train_callable_factory.initial_size == loaded.initial_size
    assert train_callable_factory.final_size == loaded.final_size
    assert train_callable_factory.warm_start == loaded.warm_start
    assert train_callable_factory.test == loaded.test


# AcquisitionFunction tests

def test_acquisition_func_get(acquisition_callable_factory, baseline, samples):
    """Unit test, not functional test"""
    acq_func = acquisition_callable_factory.get()
    assert callable(acq_func)

def test_acquistion_func_save_load(acquisition_callable_factory, instructions_dir):
    acquisition_callable_factory.save(instructions_dir)
    loaded = create_instructions.load_acquisition_func(instructions_dir)
    assert acquisition_callable_factory.baseline == loaded.baseline
    assert acquisition_callable_factory.expected_votes == loaded.expected_votes


def test_instructions_ready(instructions):
    assert instructions.ready()

def test_instructions_use_test_mode(instructions):
    instructions.use_test_mode()
    assert instructions.shards.final_size < 128

def test_instructions_save_load(instructions, instructions_dir):
    instructions.save(instructions_dir)
    loaded = create_instructions.load_instructions(instructions_dir)
    assert instructions.save_dir == loaded.save_dir
    assert instructions.subjects_per_iter == loaded.subjects_per_iter
    assert instructions.shards_per_iter == loaded.shards_per_iter
    assert instructions.n_samples == loaded.n_samples
    assert instructions.db_loc == loaded.db_loc
    # also relies on ShardConfig.save, load, tested in `make_shards_test.py`

def test_main(mocker, shard_config_loc, instructions_dir, baseline, warm_start, test):
    """Minimal unit test for correct args - individual components are tested above"""
    mocker.patch(
        'zoobot.active_learning.create_instructions.Instructions', 
        # autospec=True  # do not autospec, not sure how to also mock the shards object! TODOs
    )
    create_instructions.Instructions.shards = 'some_shards'
    mocker.patch(
        'zoobot.active_learning.create_instructions.TrainCallableFactory', 
        autospec=True
    )
    mocker.patch(
        'zoobot.active_learning.create_instructions.AcquisitionCallableFactory', 
        autospec=True
    )
    # need catalog_dir and panoptes
    create_instructions.main(shard_config_loc, catalog_dir, instructions_dir, baseline, warm_start, test, panoptes)


# Functional test for running several iterations TODO?

    # # verify the folders appear as expected
    # for iteration_n in range(active_config.iterations):
    #     # copied to iterations_test.py
    #     # separate dir for each iteration
    #     iteration_dir = os.path.join(active_config.run_dir, 'iteration_{}'.format(iteration_n))
    #     assert os.path.isdir(iteration_dir)
    #     # which has a subdir recording the estimators
    #     estimators_dir = os.path.join(iteration_dir, 'estimators')
    #     assert os.path.isdir(estimators_dir)
    #     # if initial estimator was provided, it should have been copied into the of 0th iteration subdir
    #     if iteration_n == 0:
    #         if active_config.initial_estimator_ckpt is not None:
    #             assert os.path.isdir(os.path.join(estimators_dir, active_config.initial_estimator_ckpt))
    #     else:
    #         if active_config.warm_start:
    #             # should have copied the latest estimator from the previous iteration
    #             latest_previous_estimators_dir = os.path.join(active_config.run_dir, 'iteration_{}'.format(iteration_n - 1), 'estimators')
    #             latest_previous_estimator = active_learning.get_latest_checkpoint_dir(latest_previous_estimators_dir)  # TODO double-check this func!
    #             assert os.path.isdir(os.path.join(estimators_dir, os.path.split(latest_previous_estimator)[-1]))
    
    # # read back the training tfrecords and verify they are sorted by order of mean
    # with open(active_config.train_records_index_loc, 'r') as f:
    #     acquired_shards = json.load(f)[1:]  # includes the initial shard, which is unsorted
    
    # matrix_means = []
    # for shard in acquired_shards:
    #     subjects = read_tfrecord.load_examples_from_tfrecord(
    #         [shard], 
    #         read_tfrecord.matrix_label_id_feature_spec(active_config.shards.initial_size, active_config.shards.channels)
    #     )
    #     shard_matrix_means = np.array([x['matrix'].mean() for x in subjects])

    #     # check that images have been saved to shards in monotonically decreasing order...
    #     assert all(shard_matrix_means[1:] < shard_matrix_means[:-1])
    #     matrix_means.append(shard_matrix_means)
    # # ...but not across all shards, since we only predict on some shards at a time
    # all_means = np.concatenate(matrix_means)
    # assert not all(all_means[1:] < all_means[:-1])



    # def mock_load_predictor(loc):
    #     # assumes run is configured for 3 iterations in total
    #     with open(os.path.join(loc, 'dummy_model.txt'), 'r') as f:
    #         training_records = json.load(f)
    #     if len(training_records) == 1:
    #         assert 'initial_train' in training_records[0]
    #         return 'initial train only'
    #     if len(training_records) == 2:
    #         return 'one acquired record'
    #     if len(training_records) == 3:
    #         return 'two acquired records'
    #     else:
    #         raise ValueError('More training records than expected!')
    # monkeypatch.setattr(active_learning.make_predictions, 'load_predictor', mock_load_predictor)

    # def mock_get_samples_of_subjects(model, subjects, n_samples):
    #     # only give the correct samples if you've trained on two acquired records
    #     # overrides conftest
    #     assert isinstance(subjects, list)
    #     example_subject = subjects[0]
    #     assert isinstance(example_subject, dict)
    #     assert 'matrix' in example_subject.keys()
    #     assert isinstance(n_samples, int)

    #     response = []
    #     for subject in subjects:
    #         if model == 'two acquired records':
    #             response.append([np.mean(subject['matrix'])] * n_samples)
    #         else:
    #             response.append(np.random.rand(n_samples))
    #     return np.array(response)
    # monkeypatch.setattr(active_learning.make_predictions, 'get_samples_of_subjects', mock_get_samples_of_subjects)