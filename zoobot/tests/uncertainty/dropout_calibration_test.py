import pytest

import os

import numpy as np

from zoobot.uncertainty import dropout_calibration
from zoobot.tests import TEST_FIGURE_DIR


def test_coverage_fraction(predictions, true_params):
    # here, the posterior is good, expect correct coverage
    alpha = 0.05
    coverage = dropout_calibration.coverage_fraction(predictions, true_params, alpha)
    assert 0.99 > coverage > 0.9  # sig2 should be 0.95 coverage, on average


def test_check_coverage_fractions(predictions, true_params):
    alpha_eval, coverage_fracs = dropout_calibration.check_coverage_fractions(
        predictions,
        true_params
    )


def test_visualise_calibration():
    alpha_eval = np.log10(np.logspace(0.05, 0.32))
    coverage_at_alpha = np.random.rand(len(alpha_eval))
    save_loc = os.path.join(TEST_FIGURE_DIR, 'test_visualise_calibration.png')
    dropout_calibration.visualise_calibration(alpha_eval, coverage_at_alpha, save_loc)


def test_visualise_calibration_meaningful(predictions, true_params):

    alpha_eval, coverage_fracs = dropout_calibration.check_coverage_fractions(
        predictions,
        true_params
    )
    save_loc = os.path.join(TEST_FIGURE_DIR, 'test_visualise_calibration_meaningful.png')
    dropout_calibration.visualise_calibration(alpha_eval, coverage_fracs, save_loc)

# to plot multiple series, will need to refactor the plotting
