import pytest

import os
import random

import numpy as np
import tensorflow as tf

from zoobot.tfrecord import create_tfrecord
from zoobot.estimators.estimator_params import default_four_layer_architecture, default_params
from zoobot.estimators.run_estimator import run_estimator
from zoobot.estimators.estimator_funcs import four_layer_binary_classifier


# copied from input_utils_test...

@pytest.fixture(scope='module')
def size():
    return 64


@pytest.fixture(scope='module')
def true_image_values():
    return 3.


@pytest.fixture(scope='module')
def false_image_values():
    return -3.


@pytest.fixture(scope='module')
def example_data(size, true_image_values, false_image_values):
    n_true_examples = 100
    n_false_examples = 400

    true_images = [np.ones((size, size, 3), dtype=float) * true_image_values for n in range(n_true_examples)]
    false_images = [np.ones((size, size, 3), dtype=float) * false_image_values for n in range(n_false_examples)]
    true_labels = [1 for n in range(n_true_examples)]
    false_labels = [0 for n in range(n_false_examples)]

    true_data = list(zip(true_images, true_labels))
    false_data = list(zip(false_images, false_labels))
    all_data = true_data + false_data
    random.shuffle(all_data)
    return all_data


@pytest.fixture()
def tfrecord_dir(tmpdir):
    return tmpdir.mkdir('tfrecord_dir').strpath


@pytest.fixture()
def tfrecord_train_loc(tfrecord_dir):
    return '{}/train.tfrecords'.format(tfrecord_dir)


@pytest.fixture()
def tfrecord_test_loc(tfrecord_dir):
    return '{}/test.tfrecords'.format(tfrecord_dir)


# TODO investigate how to share fixtures across test files?
@pytest.fixture()
def example_tfrecords(tfrecord_train_loc, tfrecord_test_loc, example_data):
    tfrecord_locs = [
        tfrecord_train_loc,
        tfrecord_test_loc
    ]
    for tfrecord_loc in tfrecord_locs:
        if os.path.exists(tfrecord_loc):
            os.remove(tfrecord_loc)
        writer = tf.python_io.TFRecordWriter(tfrecord_loc)

        for example in example_data:
            writer.write(create_tfrecord.serialize_image_example(matrix=example[0], label=example[1]))
        writer.close()


@pytest.fixture()
def params(tmpdir, example_tfrecords, size, tfrecord_train_loc, tfrecord_test_loc):
    params = default_params()
    params.update(default_four_layer_architecture())
    params['image_dim'] = size
    params['log_dir'] = 'runs/test_case'
    params['epochs'] = 1  # stop early, only run one train/eval cycle
    params['log_dir'] = tmpdir.mkdir('log_dir').strpath
    params['train_loc'] = tfrecord_train_loc
    params['test_loc'] = tfrecord_test_loc
    params['logging_hooks'] = None, None, None  # no train, eval or predict hooks
    return params


@pytest.fixture()
def model_fn():
    return four_layer_binary_classifier


def test_run_experiment(model_fn, params):
    run_estimator(model_fn, params)
    assert os.path.exists(params['log_dir'])
