import pytest

import os

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from zoobot.estimators.bayesian_estimator_funcs import binomial_loss
from zoobot.tests import TEST_FIGURE_DIR


# by default
# @pytest.fixture()
# def n_votes():
#     return 40

@pytest.fixture()
def single_label():
    return tf.constant(0.3, dtype=tf.float32)  # 30% vote fraction


@pytest.fixture()
def single_prediction():
    return tf.constant(0.5, dtype=tf.float32)


def test_binomial_loss_1D(single_label, single_prediction):

    neg_log_likelihood = binomial_loss(single_label, single_prediction)

    with tf.Session() as sess:
        neg_log_likelihood = sess.run([neg_log_likelihood])


def test_binomial_loss_1D_plot():

    labels = tf.placeholder(tf.float32, shape=())
    predictions = tf.placeholder(tf.float32, shape=())
    neg_log_likelihood = binomial_loss(labels, predictions)

    neg_likilihoods = []
    x_range = np.linspace(0, 1., num=100)
    y = 0.3
    for x in x_range:
        with tf.Session() as sess:
            result = sess.run(
                [neg_log_likelihood],
                feed_dict={
                    labels: y,
                    predictions: x}
                    )
            neg_likilihoods.append(result)
    
    plt.plot(x_range, neg_likilihoods, label='Neg log likelihood')
    plt.axvline(y, linestyle='--', label='True vote fraction')
    plt.xlabel('Model prediction')
    plt.ylabel('Negative log likelihood')
    plt.ylim([0., 100.])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_FIGURE_DIR, 'binomial_loss.png'))
    