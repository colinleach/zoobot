import pytest

import os
import shutil

import numpy as np

from zoobot.active_learning import iterations

@pytest.fixture()
def initial_estimator_ckpt(tmpdir):
    return tmpdir.mkdir('some_datetime_ckpt').strpath


@pytest.fixture()
def new_iteration(tmpdir, initial_estimator_ckpt):
    run_dir = tmpdir.mkdir('run_dir').strpath
    iteration_n = 0

    return iterations.Iteration(
        run_dir,
        iteration_n,
        initial_estimator_ckpt
    )


def test_init(tmpdir, initial_estimator_ckpt):
        run_dir = tmpdir.mkdir('run_dir').strpath
        iteration_n = 0

        _ = iterations.Iteration(
            run_dir,
            iteration_n,
            initial_estimator_ckpt
        )

        expected_iteration_dir = os.path.join(run_dir, 'iteration_{}'.format(iteration_n))
        assert os.path.isdir(expected_iteration_dir)

        expected_estimators_dir = os.path.join(expected_iteration_dir, 'estimators')
        assert os.path.isdir(expected_estimators_dir)

        expected_metrics_dir = os.path.join(expected_iteration_dir, 'metrics')
        assert os.path.isdir(expected_metrics_dir)

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
        return np.random.rand(n_subjects, initial_size, initial_size, 3), np.random.rand(n_subjects, n_samples)
    monkeypatch.setattr(iterations.active_learning, 'make_predictions_on_tfrecord', mock_make_predictions_on_tfrecord)

    subjects, samples = new_iteration.make_predictions(shard_locs, size)
    assert len(subjects) == 256 * len(shard_locs)
    assert samples.shape == (256 * len(shard_locs), new_iteration.n_samples)
