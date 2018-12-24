import pytest

import numpy as np

@pytest.fixture()
def typical_vote_frac():
    return 0.5


@pytest.fixture()
def typical_scatter():
    return 0.1


@pytest.fixture()
def n_subjects():
    return 1028  # more subjects decreases coverage variation between each confidence level


@pytest.fixture()
def n_samples():
    return 100  # more samples decreases systematic offset on coverage vs confidence (from pymc3)


@pytest.fixture()
def predictions(typical_vote_frac, typical_scatter, n_subjects, n_samples):
    return np.random.normal(loc=typical_vote_frac, scale=typical_scatter, size=(n_subjects, n_samples))


@pytest.fixture()
def true_params(typical_vote_frac, typical_scatter, n_subjects):
    return np.random.normal(loc=typical_vote_frac, scale=typical_scatter, size=n_subjects)
