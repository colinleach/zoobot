import os
import random

import numpy as np
import pytest
import tensorflow as tf

from zoobot.estimators.input_utils import input
from zoobot.tfrecord import create_tfrecord

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
    print('starting: probs: ', np.mean(true_labels + false_labels))

    true_data = list(zip(true_images, true_labels))
    false_data = list(zip(false_images, false_labels))
    all_data = true_data + false_data
    random.shuffle(all_data)
    return all_data


@pytest.fixture()
def tfrecord_dir(tmpdir):
    return tmpdir.mkdir('tfrecord_dir').strpath


@pytest.fixture()
def example_tfrecords(tfrecord_dir, example_data):
    tfrecord_locs = [
        '{}/train.tfrecords'.format(tfrecord_dir),
        '{}/test.tfrecords'.format(tfrecord_dir)
    ]
    for tfrecord_loc in tfrecord_locs:
        if os.path.exists(tfrecord_loc):
            os.remove(tfrecord_loc)
        writer = tf.python_io.TFRecordWriter(tfrecord_loc)

        for example in example_data:
            writer.write(create_tfrecord.serialize_image_example(matrix=example[0], label=example[1]))
        writer.close()


def test_input_utils(tfrecord_dir, example_tfrecords, size, true_image_values, false_image_values):

    # example_tfrecords sets up the tfrecords to read - needs to be an arg but is implicitly called by pytest

    train_batch = 64
    test_batch = 128

    train_loc = tfrecord_dir + '/train.tfrecords'
    test_loc = tfrecord_dir + '/test.tfrecords'
    assert os.path.exists(train_loc)
    assert os.path.exists(test_loc)

    train_features, train_labels = input(
        tfrecord_loc=train_loc,
        name='train',
        size=size,
        batch_size=train_batch,
        stratify=False,
        transform=False,  # no augmentations
        adjust=False,  # no augmentations
        channels=3
    )
    train_images = train_features['x']

    train_features_strat, train_labels_strat = input(
        tfrecord_loc=train_loc,
        name='train',
        size=size,
        batch_size=train_batch,
        stratify=True,
        init_probs=np.array([0.8, 0.2]),
        transform=False,  # no augmentations
        adjust=False,  # no augmentations
        channels=3,
    )
    train_images_strat = train_features_strat['x']

    test_features, test_labels = input(
        tfrecord_loc=test_loc,
        name='test',
        size=size,
        batch_size=test_batch,
        stratify=False,
        transform=False,  # no augmentations
        adjust=False,  # no augmentations
        channels=3
    )
    test_images = test_features['x']

    test_features_strat, test_labels_strat = input(
        tfrecord_loc=test_loc,
        name='test',
        size=size,
        batch_size=test_batch,
        stratify=True,
        init_probs=np.array([0.8, 0.2]),
        transform=False,  # no augmentations
        adjust=False,  # no augmentations
        channels=3
    )
    test_images_strat = test_features_strat['x']

    with tf.train.MonitoredSession() as sess:  # mimic Estimator environment

        train_images, train_labels = sess.run([train_images, train_labels])
        assert len(train_labels) == train_batch
        assert train_images.shape[0] == train_batch
        assert train_labels.mean() < .6  # should not be stratified
        assert train_images.shape == (train_batch, size, size, 1)
        verify_images_match_labels(train_images, train_labels, true_image_values, false_image_values, size)

        train_images_strat, train_labels_strat = sess.run([train_images_strat, train_labels_strat])
        assert len(train_labels_strat) == train_batch
        assert train_images_strat.shape[0] == train_batch
        assert train_labels_strat.mean() < 0.75 and train_labels_strat.mean() > 0.25  # stratify not very accurate...
        assert train_images_strat.shape == (train_batch, size, size, 1)
        verify_images_match_labels(train_images_strat, train_labels_strat, true_image_values, false_image_values, size)

        test_images, test_labels = sess.run([test_images, test_labels])
        assert len(test_labels) == test_batch
        assert test_images.shape[0] == test_batch
        assert test_labels.mean() < 0.6  # should not be stratified
        assert test_images.shape == (test_batch, size, size, 1)
        verify_images_match_labels(test_images, test_labels, true_image_values, false_image_values, size)

        test_images_strat, test_labels_strat = sess.run([test_images_strat, test_labels_strat])
        assert len(test_labels_strat) == test_batch
        assert test_images_strat.shape[0] == test_batch
        assert test_labels_strat.mean() < 0.75 and test_labels_strat.mean() > 0.25  # stratify not very accurate...
        assert test_images_strat.shape == (test_batch, size, size, 1)
        verify_images_match_labels(test_images_strat, test_labels_strat, true_image_values, false_image_values, size)


def verify_images_match_labels(images, labels, true_values, false_values, size):
    for example_n in range(len(labels)):
        if labels[example_n] == 1:
            expected_values = true_values
        else:
            expected_values = false_values
        expected_matrix = np.ones((size, size, 1), dtype=np.float32) * expected_values
        assert images[example_n, :, :, :] == pytest.approx(expected_matrix)
