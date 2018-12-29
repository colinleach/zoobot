import pytest

from zoobot.active_learning import check_uncertainty


def test_model_init(samples):
    example_metrics = check_uncertainty.Model(samples, labels=None, name='example')
    assert example_metrics.name == 'example'
