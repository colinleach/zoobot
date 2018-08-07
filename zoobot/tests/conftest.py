import os

from astropy.io import fits
from PIL import Image
import numpy as np
import tensorflow as tf
import pytest

from zoobot.tests import TEST_EXAMPLE_DIR


@pytest.fixture()
def size():
    return 28


@pytest.fixture()
def channels():
    return 3


@pytest.fixture()
def example_tfrecord_loc():
    loc = os.path.join(TEST_EXAMPLE_DIR, 'panoptes_featured_s28_l0.4_test.tfrecord')
    assert os.path.exists(example_tfrecord_loc)
    return loc


@pytest.fixture()
def visual_check_image(size):
    # actual image used for visual checks
    image = Image.open(os.path.join(TEST_EXAMPLE_DIR, 'example_b.png'))
    image.thumbnail((size, size))
    return tf.constant(np.array(image))


@pytest.fixture()
def n_examples():
    return 2000


@pytest.fixture()
def features(n_examples, size, channels):
    # {'feature_name':array_of_values} format expected
    return {'x': tf.constant(np.random.rand(n_examples, size, size, channels).astype(np.float32), shape=[n_examples, size, size, channels], dtype=tf.float32)}


@pytest.fixture()
def labels(n_examples):
    return tf.constant(np.random.randint(low=0, high=2, size=n_examples), shape=[n_examples], dtype=tf.int32)


@pytest.fixture()
def batch_size():
    return 10

@pytest.fixture()
def train_input_fn():
    def input_function_callable(features, labels, batch_size):
        """An input function for training
        This input function builds an input pipeline that yields batches of (features, labels) pairs,
        where features is a dictionary features.
        # https://www.tensorflow.org/get_started/datasets_quickstart
        """
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        return dataset.make_one_shot_iterator().get_next()
    return input_function_callable


@pytest.fixture()
def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        print('No labels')
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples - don't repeat though!
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()
