import pytest

import os
import shutil

from zoobot.active_learning import iterations

@pytest.fixture()
def initial_estimator_ckpt(tmpdir):
    return tmpdir.mkdir('some_datetime_ckpt').strpath


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

        # if initial estimator was provided, it should have been copied into the of 0th iteration subdir
        if initial_estimator_ckpt is not None:
            expected_initial_estimator_copy = os.path.join(expected_estimators_dir, 'some_datetime_ckpt')
            assert os.path.isdir(expected_initial_estimator_copy)
