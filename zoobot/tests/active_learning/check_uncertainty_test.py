import pytest

from zoobot.active_learning import check_uncertainty

@pytest.fixture()
def model(samples):
    return check_uncertainty.Model(samples, labels=None, name='example')


def test_model_init(samples):
    example_metrics = check_uncertainty.Model(samples, labels=None, name='example')
    assert example_metrics.name == 'example'
    # simply checks if it executes - each component is tested below


