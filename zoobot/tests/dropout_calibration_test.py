import pytest

import numpy as np

from zoobot.uncertainty import dropout_calibration


def test_verify_uncertainty(monkeypatch):

    model = None
    subjects = np.random.rand(20, 128, 128)
    true_param = 0.5  # internal only
    true_params = np.ones(len(subjects)) * true_param
    n_samples = 1000

    def mock_predictions(model, subjects, n_samples):
        return np.random.normal(loc=0.5, scale=0.1, size=(len(subjects), n_samples))

    monkeypatch.setattr(
        dropout_calibration.make_predictions, 
        'get_samples_of_subjects', 
        mock_predictions
    )
    coverage = dropout_calibration.verify_uncertainty(model, subjects, true_params, n_samples)
    assert 0.99 < coverage < 0.9  # sig2 should be 0.95 coverage, on average
