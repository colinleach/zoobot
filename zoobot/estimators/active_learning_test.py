import pytest

import logging
import os
import random

import numpy as np
import tensorflow as tf
import pandas as pd

from zoobot.tfrecord import create_tfrecord
from zoobot.estimators.architecture_values import default_four_layer_architecture, default_params
from zoobot.estimators.run_estimator import four_layer_regression_classifier

from zoobot.estimators import active_learning, dummy_estimator

logging.basicConfig(
    filename='active_learning_test.log',
    filemode='w',
    format='%(asctime)s %(message)s',
    level=logging.DEBUG)


@pytest.fixture()
def known_subjects():
    data = [{
        'some_feature': np.random.rand(1),
        'label': np.random.randint(0, 2, size=1)
    }]
    return pd.DataFrame(data)


@pytest.fixture()
def unknown_subjects():
    data = [{
        'some_feature': np.random.rand(1),
    }]
    return pd.DataFrame(data)


@pytest.fixture()
def test_dir(tmpdir):
    return tmpdir.strpath


@pytest.fixture()
def params(test_dir):
    params = active_learning.get_active_learning_params()
    params['known_tfrecord_loc'] = test_dir + 'known.tfrecord'
    params['unknown_tfrecord_loc'] = test_dir + 'unknown.tfrecord'


def test_run_experiment(estimator, params, known_subjects, unknown_subjects):
    active_learning.run_active_learning(estimator, params, known_subjects, unknown_subjects)
    assert os.path.exists(params['log_dir'])
