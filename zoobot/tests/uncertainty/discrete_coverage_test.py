import pytest

import os

import scipy
import numpy as np

from zoobot.estimators import make_predictions
from zoobot.uncertainty import discrete_coverage
from zoobot.tests import TEST_FIGURE_DIR


@pytest.fixture()
def n_subjects():
    return 24

@pytest.fixture()
def n_draws():
    return 40

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


def test_evaluate_discrete_coverage(volunteer_votes, bin_prob_of_samples_by_k):
    # if I'm clever, I can get error bars
    # df of form: [max +/- n states, mean observed frequency, mean probability prediction]
    coverage_df = discrete_coverage.evaluate_discrete_coverage(volunteer_votes, bin_prob_of_samples_by_k)
    save_loc = os.path.join(TEST_FIGURE_DIR, 'discrete_coverage.png')
    discrete_coverage.plot_coverage_df(coverage_df, save_loc)
