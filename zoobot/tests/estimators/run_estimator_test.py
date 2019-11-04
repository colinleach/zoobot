import pytest

import os
import random

import numpy as np
import tensorflow as tf

from zoobot.tfrecord import create_tfrecord
from zoobot.estimators.estimator_params import default_four_layer_architecture, default_params
from zoobot.estimators import run_estimator, estimator_funcs, bayesian_estimator_funcs, dummy_image_estimator, losses
from zoobot.active_learning import default_estimator_params


@pytest.fixture()
def size():
    return 64  # override to be a bit bigger

@pytest.fixture()
def log_dir(tmpdir):
    return tmpdir.mkdir('log_dir').strpath  # also includes saved model


@pytest.fixture()
def n_examples():
    return 128  # possible error restoring model if this is not exactly one batch?


@pytest.fixture()
def features(n_examples):
    # {'feature_name':array_of_values} format expected
    feature_shape = [n_examples, 28, 28, 1]
    return {
        'x': tf.constant(
            np.random.rand(*feature_shape), 
            shape=feature_shape, 
            dtype=tf.float32)
        }


@pytest.fixture
def labels(request, n_examples):
    if request.param == 'continuous':
        return tf.constant(
            np.random.uniform(size=n_examples),
            shape=[n_examples],
            dtype=tf.float32)


@pytest.fixture
def fake_dataset(size, channels):
    dataset_len = 1000
    batch_dim = 16
    features = np.random.rand(dataset_len, size, size, channels).astype(np.float32)
    true_labels = np.array([np.random.choice([0, 1, 2, 3, 4]) for n in range(dataset_len)]).astype(np.float32)
    false_labels = 4 - true_labels
    labels = np.stack((true_labels, false_labels), axis=1)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.shuffle(dataset_len).batch(batch_dim)


@pytest.fixture()
def model(size):
    model = bayesian_estimator_funcs.BayesianModel(
        output_dim=2,
        conv1_filters=4,
        conv1_kernel=3,
        conv2_filters=8,
        conv2_kernel=3,
        conv3_filters=16,
        conv3_kernel=3,
        dense1_units=16,
        dense1_dropout=0.5,
        predict_dropout=0.5,  # change this to calibrate
        regression=True,  # important!
        log_freq=10,
        image_dim=size,
    )
    model.compile(
        loss=losses.multinomial_loss,
        optimizer=tf.keras.optimizers.Adam()
    )
    return model

@pytest.fixture()
def run_config(size, channels, model, log_dir):
    config = default_estimator_params.RunEstimatorConfig(
        initial_size=size,
        final_size=size,
        channels=channels,
        label_cols=['label_a', 'label_b'],
        log_dir=log_dir
    )
    config.model = model
    config.epochs = 2
    return config


def test_run_experiment(
    run_config,
    fake_dataset,
    monkeypatch):

    # TODO need to test estimator with input functions!
    # mock both input functions
    def dummy_input(config=None):
        return fake_dataset
    monkeypatch.setattr(run_estimator.input_utils, 'get_input', dummy_input)
    monkeypatch.setattr(run_config, 'is_ready_to_train', lambda: True)

    run_estimator.run_estimator(run_config)
    assert os.path.exists(run_config.log_dir)
