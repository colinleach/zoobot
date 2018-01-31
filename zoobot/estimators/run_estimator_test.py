import pytest

import os

from zoobot.estimators.architecture_values import default_four_layer_architecture, default_params
from zoobot.estimators.run_estimator import run_experiment, four_layer_binary_classifier


@pytest.fixture()
def params(tmpdir):
    params = default_params()
    params.update(default_four_layer_architecture())
    params['image_dim'] = 64
    params['log_dir'] = 'runs/chollet_128_triple'
    params['epochs'] = 1  # stop early, only run one train/eval cycle
    params['log_dir'] = tmpdir.mkdir('log_dir').strpath
    return params


@pytest.fixture()
def model_fn():
    return four_layer_binary_classifier


def test_run_experiment(model_fn, params):
    run_experiment(model_fn, params)
    assert os.path.exists(params['log_dir'])
