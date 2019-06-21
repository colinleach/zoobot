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


@pytest.fixture(params=[{'labels': None}, {'labels': 'some'}])
def model(samples, request):
    if request.param['labels'] is None:
        return metrics.Model(samples, labels=None, name='example_no_labels')
    else:
        return metrics.Model(samples, labels=np.random.rand(len(samples)), name='example')


def test_model_init(samples):
    example_metrics = metrics.Model(samples, labels=None, name='example')
    assert example_metrics.name == 'example'
    # simply checks if it executes - each component is tested below


def test_show_acquisitions_vs_predictions(model, save_dir):
    model.show_acquisitions_vs_predictions(save_dir)
    assert os.path.exists(os.path.join(save_dir, 'entropy_by_prediction.png'))


def test_show_coverage(model, save_dir):
    if model.labels is None:
        with pytest.raises(ValueError):
            model.show_coverage(save_dir)
    else:
        model.show_coverage(save_dir)
        assert os.path.exists(os.path.join(save_dir, 'discrete_coverage.png'))
