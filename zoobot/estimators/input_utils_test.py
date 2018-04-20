import os
import random

import numpy as np
import pytest
import tensorflow as tf

from zoobot.estimators.input_utils import input
from zoobot.tfrecord.create_tfrecord import image_to_tfrecord

TEST_EXAMPLE_DIR = 'zoobot/test_examples'

@pytest.fixture(scope='module')
def size():
    return 4


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


@pytest.fixture(scope='module')
def example_tfrecords(all_data, tmpdir):
    tfrecord_dir = tmpdir.mkdir('tfrecord_dir').strpath
    tfrecord_locs = [
        '{}/train.tfrecords'.format(tfrecord_dir),
        '{}/test.tfrecords'.format(tfrecord_dir)
    ]
    for tfrecord_loc in tfrecord_locs:
        if os.path.exists(tfrecord_loc):
            os.remove(tfrecord_loc)
        writer = tf.python_io.TFRecordWriter(tfrecord_loc)

        for example in all_data:
            image_to_tfrecord(matrix=example[0], label=example[1], writer=writer)
        writer.close()


def test_input_utils_stratified(size, true_image_values, false_image_values):

    train_batch = 64
    test_batch = 128

    train_loc = TEST_EXAMPLE_DIR + '/train.tfrecords'
    test_loc = TEST_EXAMPLE_DIR + '/test.tfrecords'
    assert os.path.exists(train_loc)
    assert os.path.exists(test_loc)

    train_features, train_labels = input(
        tfrecord_loc=TEST_EXAMPLE_DIR + '/train.tfrecords',
        name='train',
        size=size,
        batch=train_batch,
        stratify=True,
        transform=False,  # no augmentations
        adjust=False  # no augmentations
    )
    train_images = train_features['x']

    test_features, test_labels = input(
        tfrecord_loc=TEST_EXAMPLE_DIR + '/' + 'test.tfrecords',
        name='test',
        size=size,
        batch=test_batch,
        stratify=True,
        transform=False,  # no augmentations
        adjust=False  # no augmentations
    )
    test_images = test_features['x']

    with tf.train.MonitoredSession() as sess:  # mimic Estimator environment

        train_images, train_labels = sess.run([train_images, train_labels])
        assert len(train_labels) == train_batch
        assert train_images.shape[0] == train_batch
        assert train_labels.mean() < 0.75 and train_labels.mean() > 0.25  # stratify not very accurate...
        assert train_images.shape == (train_batch, size, size, 1)
        verify_images_match_labels(train_images, train_labels, true_image_values, false_image_values, size)

        test_images, test_labels = sess.run([test_images, test_labels])
        assert len(test_labels) == test_batch
        assert test_images.shape[0] == test_batch
        assert test_labels.mean() < 0.75 and test_labels.mean() > 0.25  # stratify not very accurate...
        assert test_images.shape == (test_batch, size, size, 1)
        verify_images_match_labels(test_images, test_labels, true_image_values, false_image_values, size)


def verify_images_match_labels(images, labels, true_values, false_values, size):
    for example_n in range(len(labels)):
        if labels[example_n] == 1:
            expected_values = true_values
        else:
            expected_values = false_values
        expected_matrix = np.ones((size, size, 1), dtype=np.float32) * expected_values
        assert images[example_n, :, :, :] == pytest.approx(expected_matrix)
