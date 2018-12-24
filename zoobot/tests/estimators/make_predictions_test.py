import pytest

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zoobot.tests import TEST_FIGURE_DIR
from zoobot.estimators import make_predictions


@pytest.fixture
def predictions():
    return np.array([[0.4, 0.5, 0.6] for n in range(2)])


@pytest.fixture
def n_draws():
    return 10


@pytest.fixture
def bin_probs(n_draws):
    unscaled_probs = np.random.rand(12, 3, n_draws + 1)
    total_by_k = np.sum(unscaled_probs, axis=2)
    total_by_k_expanded = np.tile(np.expand_dims(total_by_k, axis=-1), n_draws + 1)
    return unscaled_probs / total_by_k_expanded


def test_load_predictor(predictor_model_loc):
    predictor = make_predictions.load_predictor(predictor_model_loc)
    assert callable(predictor)


def test_get_samples_of_subjects(predictor, parsed_example):
    n_subjects = 10
    n_samples = 5
    subjects = [parsed_example for n in range(n_subjects)]
    samples = make_predictions.get_samples_of_subjects(predictor, subjects, n_samples)
    assert samples.shape == (10, 5)


def test_binomial_prob_per_k(n_draws):
    sampled_rho = 0.5
    prob_per_k = make_predictions.binomial_prob_per_k(sampled_rho, n_draws)
    for n in range(int(n_draws/2) - 1):
        assert np.allclose(prob_per_k[n], prob_per_k[-1-n])
    assert prob_per_k[0] < prob_per_k[1] < prob_per_k[2]


def test_predictive_binomial_entropy(bin_probs):
    predictive_entropy = make_predictions.predictive_binomial_entropy(bin_probs)
    assert predictive_entropy.ndim == 1
    assert not np.allclose(predictive_entropy, make_predictions.expected_binomial_entropy(bin_probs))



def test_expected_binomial_entropy(bin_probs):
    entropy = make_predictions.expected_binomial_entropy(bin_probs)
    assert entropy.ndim == 1
    assert entropy.min() > 0


def test_predictive_and_expected_entropy_functional():
    predictions = np.random.rand(100, 30)
    bin_probs = make_predictions.bin_prob_of_samples(predictions, n_draws=40)
    predictive_entropy = make_predictions.predictive_binomial_entropy(bin_probs)
    expected_entropy = make_predictions.expected_binomial_entropy(bin_probs)
    mutual_info = predictive_entropy - expected_entropy
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
    mean_prediction = predictions.mean(axis=1)
    ax0.scatter(mean_prediction, predictive_entropy, label='Predictive')
    ax0.set_xlabel('Mean prediction')
    ax0.set_ylabel('Predictive Entropy')
    ax1.scatter(mean_prediction, expected_entropy, label='Expected')
    ax1.set_xlabel('Mean prediction')
    ax1.set_ylabel('Expected Entropy')
    ax2.scatter(mean_prediction, mutual_info, label='Mutual Info')
    ax2.set_xlabel('Mean prediction')
    ax2.set_ylabel('Mutual Information')
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'mutual_info.png'))



# def test_binomial_entropy():
#     n_draws = 10
#     rho = 0.5
#     entropy = make_predictions.binomial_entropy(rho, n_draws)
#     assert entropy.shape == ()  # should be scalar


# def test_binomial_entropy_vectorized():
#     n_draws = 10  # not yet tested with varying n
#     rho = [0.1, 0.5, 0.9]
#     entropy = make_predictions.binomial_entropy(rho, n_draws)
#     assert entropy.min() > 0
#     assert len(entropy) == 3  # should be scalar
#     assert entropy[1] == entropy.max()
#     assert np.allclose(entropy[0], entropy[-1])


# def test_binomial_entropy_plotted():
#     pass  # TODO