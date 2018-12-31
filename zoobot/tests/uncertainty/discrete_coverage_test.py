import pytest

import os

import scipy
import numpy as np
import pandas as pd

from zoobot.estimators import make_predictions
from zoobot.uncertainty import discrete_coverage
from zoobot.tests import TEST_FIGURE_DIR


@pytest.fixture()
def n_subjects():
    return 24

@pytest.fixture()
def n_draws():
    return 20

@pytest.fixture()
def true_p(n_subjects):
    return np.random.rand(n_subjects)  # per subject

@pytest.fixture()
def volunteer_votes(true_p, n_draws):
    return np.concatenate([scipy.stats.binom.rvs(p=true_p[subject_n], n=n_draws, size=1) for subject_n in range(len(true_p))])


@pytest.fixture()
def bin_prob_of_samples_by_k(n_subjects, n_draws, true_p):
    """ of form [subject_n, sample_n, k] """
    # return make_predictions.bin_prob_of_samples(samples, n_draws=40)
    n_samples = 10
    bin_probs = np.zeros((n_subjects, n_samples, n_draws + 1))
    for subject_n in range(n_subjects):
        for sample_n in range(n_samples):
            for k in range(n_draws):
                bin_probs[subject_n, sample_n, k] = scipy.stats.binom(p=true_p[subject_n], n=n_draws).pmf(k)
    return bin_probs


@pytest.fixture()
def coverage_df():
    return pd.DataFrame([
        {
            'max_state_error': 4,
            'probability': 0.7,
            'observed': False
        },
        {
            'max_state_error': 4,
            'probability': 1.,
            'observed': True
        },
        {
            'max_state_error': 12,
            'probability': 0.3,
            'observed': False
        },
        {
            'max_state_error': 12,
            'probability': 0.,
            'observed': True
        },
        {
            'max_state_error': 12,
            'probability': 0.4,
            'observed': False
        },
        {
            'max_state_error': 12,
            'probability': 1.,
            'observed': True
        },
    ])

def test_evaluate_discrete_coverage(volunteer_votes, bin_prob_of_samples_by_k):
    # if I'm clever, I can get error bars
    # df of form: [max +/- n states, mean observed frequency, mean probability prediction]
    coverage_df = discrete_coverage.evaluate_discrete_coverage(volunteer_votes, bin_prob_of_samples_by_k)
    assert np.allclose(len(coverage_df), np.product(bin_prob_of_samples_by_k) * 10 * 2)  # 10 test errors, observed Y/N
    save_loc = os.path.join(TEST_FIGURE_DIR, 'discrete_coverage.png')
    discrete_coverage.plot_coverage_df(coverage_df, save_loc)



def test_reduce_coverage_df(coverage_df):
    reduced_df = discrete_coverage.reduce_coverage_df(coverage_df)
    assert len(reduced_df) == 2
    first_row = reduced_df.iloc[0]
    assert first_row['max_state_error'] == 4
    assert np.allclose(first_row['prediction'], 0.7)
    assert np.allclose(first_row['frequency'], 1.)

    second_row = reduced_df.iloc[1]
    assert second_row['max_state_error'] == 12
    assert np.allclose(second_row['prediction'], 0.35)
    assert np.allclose(second_row['frequency'], .5)


def test_calibrate_predictions(reduced_df):
    calibrated_df = discrete_coverage.calibrate_predictions(reduced_df)
    # TODO