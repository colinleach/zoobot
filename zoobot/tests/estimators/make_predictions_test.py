import pytest

import numpy as np

from zoobot.estimators import make_predictions


def test_load_predictor(predictor_model_loc):
    predictor = make_predictions.load_predictor(predictor_model_loc)
    assert callable(predictor)


def test_get_samples_of_subjects(predictor, parsed_example):
    n_subjects = 10
    n_samples = 5
    subjects = [parsed_example for n in range(n_subjects)]
    samples = make_predictions.get_samples_of_subjects(predictor, subjects, n_samples)
    assert samples.shape == (10, 5)


def test_binomial_prob_per_k():
    n_draws = 10
    sampled_rho = 0.5
    prob_per_k = make_predictions.binomial_prob_per_k(sampled_rho, n_draws)
    for n in range(int(n_draws/2) - 1):
        # allclose is very sensitive by default
        assert np.allclose(prob_per_k[n], prob_per_k[-1-n], atol=1e-5)
    assert prob_per_k[0] < prob_per_k[1] < prob_per_k[2]


def test_binomial_entropy():
    n_draws = 10
    rho = 0.5
    entropy = make_predictions.binomial_entropy(rho, n_draws)
    assert entropy.shape == ()  # should be scalar


def test_binomial_entropy_vectorized():
    n_draws = 10  # not yet tested with varying n
    rho = [0.1, 0.5, 0.9]
    entropy = make_predictions.binomial_entropy(rho, n_draws)
    assert entropy.min() > 0
    assert len(entropy) == 3  # should be scalar
    assert entropy[1] == entropy.max()
    assert np.allclose(entropy[0], entropy[-1])


def test_binomial_entropy_plotted():
    pass  # TODO


# def test_binomial_prob_per_k_vectorized():
# doesn't work: can't expand a dimension
#     n_draws = [10, 12]
#     sampled_rho = [0.5, 0.]
#     probs_per_k = make_predictions.binomial_prob_per_k(sampled_rho, n_draws)
    # for n in range(int(n_draws/2) - 1):
    #     # allclose is very sensitive by default
    #     assert np.allclose(prob_per_k[n], prob_per_k[-1-n], atol=1e-5)
    # assert prob_per_k[0] < prob_per_k[1] < prob_per_k[2]


# def test_predictive_binomial_entropy_1D():
#     n_draws = 10
#     sampled_rhos = np.array([0.4, 0.5, 0.6])
#     predictive_entropy = make_predictions.predictive_binomial_entropy(sampled_rhos, n_draws)
#     assert predictive_entropy.shape == ()  # many rhos should give scalar entropy
#     assert not np.allclose(predictive_entropy, make_predictions.binomial_entropy(np.mean(sampled_rhos), n_draws))
#     # assert predictive_entropy


def test_predictive_binomial_entropy_2D():
    n_draws = 10
    sampled_rhos = np.array([[0.4, 0.5, 0.6] for n in range(2)])
    assert sampled_rhos.shape == (2, 3)  # 2 subjects, 3 samples of each subject
    predictive_entropy = make_predictions.predictive_binomial_entropy(sampled_rhos, n_draws)
    assert len(predictive_entropy) == 2  # 2D rhos should give vector entropy
    assert not np.allclose(predictive_entropy, make_predictions.binomial_entropy(np.mean(sampled_rhos), n_draws))
    # assert predictive_entropy


def test_predictive_binomial_entropy_plotted():
    pass