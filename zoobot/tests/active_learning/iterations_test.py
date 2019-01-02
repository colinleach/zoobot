import pytest

import os
import shutil

import numpy as np

from zoobot.active_learning import iterations

@pytest.fixture()
def initial_estimator_ckpt(tmpdir):
    return tmpdir.mkdir('some_datetime_ckpt').strpath


@pytest.fixture()
def new_iteration(tmpdir, initial_estimator_ckpt, active_config_ready):
        run_dir = active_config_ready.run_dir
        iteration_n = 0
        prediction_shards = ['some', 'shards']

        iteration = iterations.Iteration(
            run_dir,
            iteration_n,
            prediction_shards,
            initial_db_loc=active_config_ready.db_loc,
            initial_train_tfrecords=[active_config_ready.shards.train_tfrecord_loc],
            train_callable=np.random.rand,
            acquisition_func=np.random.rand,
            n_samples=10,  # may need more samples?
            n_subjects_to_acquire=50,
            initial_size=64,
            initial_estimator_ckpt=None
        )

        return iteration


def test_init(tmpdir, initial_estimator_ckpt, active_config_ready):
        run_dir = active_config_ready.run_dir
        iteration_n = 0
        prediction_shards = ['some', 'shards']

        iteration = iterations.Iteration(
            run_dir,
            iteration_n,
            prediction_shards,
            initial_db_loc=active_config_ready.db_loc,
            initial_train_tfrecords=[active_config_ready.shards.train_tfrecord_loc],
            train_callable=np.random.rand,
            acquisition_func=np.random.rand,
            n_samples=10,  # may need more samples?
            n_subjects_to_acquire=50,
            initial_size=64,
            initial_estimator_ckpt=initial_estimator_ckpt
        )

        assert iteration.acquired_tfrecord is None
 
        expected_iteration_dir = os.path.join(run_dir, 'iteration_{}'.format(iteration_n))
        assert os.path.isdir(expected_iteration_dir)

        expected_estimators_dir = os.path.join(expected_iteration_dir, 'estimators')
        assert os.path.isdir(expected_estimators_dir)

        expected_metrics_dir = os.path.join(expected_iteration_dir, 'metrics')
        assert os.path.isdir(expected_metrics_dir)

        expected_db_loc = os.path.join(expected_iteration_dir, 'iteration.db')
        assert os.path.exists(expected_db_loc)

        # if initial estimator was provided, it should have been copied into the of 0th iteration subdir
        if initial_estimator_ckpt is not None:
            expected_initial_estimator_copy = os.path.join(expected_estimators_dir, 'some_datetime_ckpt')
            assert os.path.isdir(expected_initial_estimator_copy)


def test_get_latest_model(monkeypatch, new_iteration):
    def mock_get_latest_checkpoint(base_dir):
        return 'latest_ckpt'
    monkeypatch.setattr(iterations.active_learning, 'get_latest_checkpoint_dir', mock_get_latest_checkpoint)

    def mock_load_predictor(predictor_loc):
        if predictor_loc == 'latest_ckpt':
            return 'loaded latest model'
        else:
            return 'loaded another model'
    monkeypatch.setattr(iterations.make_predictions, 'load_predictor', mock_load_predictor)

    assert new_iteration.get_latest_model() == 'loaded latest model'


def test_make_predictions(monkeypatch, shard_locs, size, new_iteration):
    def mock_get_latest_model(self):
        return 'loaded latest model'
    monkeypatch.setattr(iterations.Iteration, 'get_latest_model', mock_get_latest_model)

    def mock_make_predictions_on_tfrecord(shard_locs, predictor, initial_size, n_samples):
        assert isinstance(shard_locs, list)
        assert predictor == 'loaded latest model'
        assert isinstance(initial_size, int)
        assert isinstance(n_samples, int)
        n_subjects = 256 * len(shard_locs)
        subjects = [{'matrix': np.random.rand(initial_size, initial_size, 3), 'id_str': str(n)} 
        for n in range(n_subjects)]
        samples = np.random.rand(n_subjects, n_samples)
        return subjects, samples
    monkeypatch.setattr(iterations.active_learning, 'make_predictions_on_tfrecord', mock_make_predictions_on_tfrecord)

    subjects, samples = new_iteration.make_predictions(shard_locs, size)
    assert len(subjects) == 256 * len(shard_locs)
    assert samples.shape == (256 * len(shard_locs), new_iteration.n_samples)


def test_save_metrics():
    pass


def test_get_train_records(new_iteration, active_config_ready):
    assert new_iteration.get_train_records() == new_iteration.initial_train_tfrecords
    new_iteration.acquired_tfrecord = 'acquired.tfrecord'
    assert new_iteration.get_train_records() == new_iteration.initial_train_tfrecords + ['acquired.tfrecord']


def test_run():
    pass