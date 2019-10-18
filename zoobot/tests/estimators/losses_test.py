import pytest

import os

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from zoobot.estimators import losses, make_predictions
from zoobot.tests import TEST_FIGURE_DIR


@pytest.fixture()
def single_label():
    return tf.constant([[6, 10], [6, 50]], dtype=tf.float32)  # 30% vote fraction


@pytest.fixture()
def single_prediction():
    return tf.constant([0.5, 0.5], dtype=tf.float32)


def test_binomial_loss(single_label, single_prediction):

    neg_log_likelihood = losses.binomial_loss(single_label, single_prediction)

    with tf.Session() as sess:
        [neg_log_likelihood] = sess.run([neg_log_likelihood])
    
    assert neg_log_likelihood[0] < neg_log_likelihood[1] # first is less improbable than second


def test_binomial_loss_1D_plot():
    # verify that np and tf versions of binomial loss look good and agree
    batch_dim = 100
    true_prob = 0.3
    n_trials = 10
    successes = n_trials * true_prob
    labels_data = np.array([[successes, n_trials] for _ in range(batch_dim)])
    predictions_data = np.linspace(0, 1., num=batch_dim) 
    labels = tf.cast(tf.constant(labels_data), tf.float32)
    predictions = tf.cast(tf.constant(predictions_data), tf.float32)
    tf_neg_log_likelihood = losses.binomial_loss(labels, predictions)
    with tf.Session() as sess:
        tf_neg_log_likelihood = sess.run(tf_neg_log_likelihood)

    np_neg_likilihoods = - make_predictions.binomial_likelihood(true_prob, predictions_data, n_trials=n_trials)
    
    plt.plot(predictions_data, np_neg_likilihoods, label='np neg log likelihood')
    plt.plot(predictions_data, tf_neg_log_likelihood, label='tf neg log likelihood')
    plt.axvline(true_prob, linestyle='--', label='True vote fraction')
    plt.xlabel('Model prediction')
    plt.ylabel('Negative log likelihood')
    plt.ylim([0., 100.])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_FIGURE_DIR, 'binomial_loss.png'))

"""Multinomial loss"""


def test_multinomial_loss():

    n_trials = 10
    batch_dim = 100
    q1_true_prob = 0.3
    true_probs_by_q = np.array([q1_true_prob, 1-q1_true_prob])
    n_answers = len(true_probs_by_q)  # only used to construct useful data, not in loss itself
    successes_by_q = n_trials * true_probs_by_q
    # successes = np.concatenate(
    #     [np.random.binomial(n=n_trials, p=true_probs_by_q[n], size=batch_dim) for n in range(n_answers)]
    # )
    successes = np.stack(
        [[successes_by_q[n]] * batch_dim for n in range(n_answers)],
        axis=1
    )
    # print(successes.shape)
    print(successes)
    # predictions are 0, 0, 0. ....1, 1, 1, same for all answers, over batch dim
    prediction_range = np.linspace(0., 1., num=batch_dim)
    predictions = np.stack(
        [prediction_range, 1-prediction_range],
        axis=1
    )
    print(predictions)
    # print(predictions.shape)

    neg_log_likelihood = losses.multinomial_loss(successes, predictions)
    with tf.Session() as sess:
        [neg_log_likelihood] = sess.run([neg_log_likelihood])

    print(neg_log_likelihood)

    plt.clf()
    plt.plot(prediction_range, neg_log_likelihood)
    plt.xlabel('Prediction for both questions')
    plt.ylabel('Neg log likelihood')
    plt.savefig(os.path.join(TEST_FIGURE_DIR, 'multinomial_loss.png'))

    # assert neg_log_likelihood[0] < neg_log_likelihood[1] # first is less improbable than second
    # assert False
