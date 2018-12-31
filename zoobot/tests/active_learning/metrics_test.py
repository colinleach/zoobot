import pytest

from zoobot.active_learning import metrics
from zoobot.tests import TEST_FIGURE_DIR


@pytest.fixture()
def model(samples):
    return metrics.Model(samples, labels=None, name='example')


def test_model_init(samples):
    example_metrics = metrics.Model(samples, labels=None, name='example')
    assert example_metrics.name == 'example'
    # simply checks if it executes - each component is tested below


def test_show_acquisitions_vs_predictions(model):
    model.show_acquisitions_vs_predictions(save_dir=TEST_FIGURE_DIR)
