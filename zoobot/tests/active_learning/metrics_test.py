import pytest

import os

import numpy as np

from zoobot.active_learning import metrics
from zoobot.tests import TEST_FIGURE_DIR

@pytest.fixture()
def save_dir():
    save_dir = os.path.join(TEST_FIGURE_DIR, 'metrics')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    return save_dir


@pytest.fixture()
def state(samples, acquisitions, id_strs):
    return metrics.IterationState(samples, acquisitions, id_strs)


@pytest.fixture()
def iteration_dir(tmpdir):
    return tmpdir.mkdir('iteration_dir').strpath

def test_save_iteration_state(iteration_dir, subjects, samples, acquisitions):
    metrics.save_iteration_state(iteration_dir, subjects, samples, acquisitions)
    assert os.path.isfile(os.path.join(iteration_dir, 'state.pickle'))


def test_load_iteration_state(subjects, samples, acquisitions, iteration_dir):
    # assumes saving works correctly 
    metrics.save_iteration_state(iteration_dir, subjects, samples, acquisitions)
    state = metrics.load_iteration_state(iteration_dir)
    assert isinstance(state, metrics.IterationState)
    # TODO should probably test with consistent id str etc

@pytest.fixture()
def model(state, request):
    return metrics.Model(state, name='example')


def test_model_init(state):
    example_metrics = metrics.Model(state, name='example')
    assert example_metrics.name == 'example'
    # simply checks if it executes - each component is tested below


def test_show_mutual_info_vs_predictions(model, save_dir):
    model.show_mutual_info_vs_predictions(save_dir)
    assert os.path.exists(os.path.join(save_dir, 'entropy_by_prediction.png'))

def test_acquisitions_vs_mean_prediction(model, save_dir):
    n_acquired = 20
    if model.acquisitions is None:
        with pytest.raises(ValueError):
            model.acquisitions_vs_mean_prediction(n_acquired, save_dir)
    else:
        model.acquisitions_vs_mean_prediction(n_acquired, save_dir)
        assert os.path.exists(os.path.join(save_dir, 'discrete_coverage.png'))
