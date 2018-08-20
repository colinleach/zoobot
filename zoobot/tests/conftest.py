import os
import random

from astropy.io import fits
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import pytest

from zoobot.tfrecord import create_tfrecord
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
    assert os.path.exists(loc)
    return loc


@pytest.fixture()
def visual_check_image_data(size):
    # actual image used for visual checks
    image = Image.open(os.path.join(TEST_EXAMPLE_DIR, 'example_b.png'))
    image.thumbnail((size, size))
    return np.array(image)


@pytest.fixture()
def visual_check_image(visual_check_image_data):
    return tf.constant(visual_check_image_data)


@pytest.fixture()
def n_examples():
    return 2000


@pytest.fixture()
def random_features(n_examples, size, channels):
    # {'feature_name':array_of_values} format expected
    return {'x': tf.constant(np.random.rand(n_examples, size, size, channels).astype(np.float32), shape=[n_examples, size, size, channels], dtype=tf.float32)}


@pytest.fixture()
def random_labels(n_examples):
    return tf.constant(np.random.randint(low=0, high=2, size=n_examples), shape=[n_examples], dtype=tf.int32)


@pytest.fixture()
def parsed_example(visual_check_image_data):
    return {
        'matrix': np.array(visual_check_image_data).flatten(),  # Parsed matrix is a 1D vector, needs reshaping
        'label': 1
    }


@pytest.fixture()
def batch_size():
    return 10


@pytest.fixture()
def train_input_fn():
    def input_function_callable(features, labels, batch_size):  # normal args, not fixtures
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
def eval_input_fn(random_features, random_labels, batch_size):
    """An input function for evaluation or prediction"""
    random_features = dict(random_features)
    if random_labels is None:
        # No labels, use only features.
        inputs = random_features
    else:
        inputs = (random_features, random_labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples - don't repeat though!
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()



@pytest.fixture(scope='module')
def true_image_values():
    return 3.


@pytest.fixture(scope='module')
def false_image_values():
    return -3.



@pytest.fixture()
def stratified_data(true_image_values, false_image_values, size, channels):
    # tuples of (image, label), with image values according to the label
    # useful to write to stratified tfrecord for input utils stratify testing
    n_true_examples = 100
    n_false_examples = 400

    true_images = [np.ones((size, size, channels), dtype=float) * true_image_values for n in range(n_true_examples)]
    false_images = [np.ones((size, size, channels), dtype=float) * false_image_values for n in range(n_false_examples)]
    true_labels = [1 for n in range(n_true_examples)]
    false_labels = [0 for n in range(n_false_examples)]

    true_data = list(zip(true_images, true_labels))
    false_data = list(zip(false_images, false_labels))
    all_data = true_data + false_data
    random.shuffle(all_data)  # inplace
    return all_data


@pytest.fixture()
def tfrecord_dir(tmpdir):
    return tmpdir.mkdir('tfrecord_dir').strpath


@pytest.fixture()
def stratified_tfrecord_locs(tfrecord_dir, stratified_data):
    tfrecord_locs = [
        os.path.join(tfrecord_dir, 'stratified_train.tfrecords'),
        os.path.join(tfrecord_dir, 'stratified_test.tfrecords')
    ]

    for tfrecord_loc in tfrecord_locs:
        if os.path.exists(tfrecord_loc):
            os.remove(tfrecord_loc)

        writer = tf.python_io.TFRecordWriter(tfrecord_loc)
        for example in stratified_data:  # depends on tfrecord.create_tfrecord
            writer.write(create_tfrecord.serialize_image_example(matrix=example[0], label=example[1]))
        writer.close()

    return tfrecord_locs  # of form [train_loc, test_loc]


@pytest.fixture()
def predictor_model_loc():  # not yet on github
    return os.path.join(TEST_EXAMPLE_DIR, 'example_saved_model/1530286779')


@pytest.fixture()
def predictor():
    return lambda x: np.random.rand()  # whatever is passed in, return a single random float score


@pytest.fixture()
def label_col():
    return 't04_spiral_a08_spiral_weighted_fraction'


@pytest.fixture()
def id_col():
    return 'id'


@pytest.fixture
def columns_to_save(label_col, id_col):
    return [
        label_col,
        id_col,  # Will break tests, saving string features is not yet implemented!
        'ra',
        'dec']


@pytest.fixture
def catalog(label_col, id_col):

    zoo1 = {
        label_col: 0.4,
        id_col: 'zoo1',
        'ra': 12.0,
        'dec': -1.0,
        'png_loc': '{}/example_a.png'.format(TEST_EXAMPLE_DIR),
        'fits_loc': '{}/example_a.fits'.format(TEST_EXAMPLE_DIR),
        'png_ready': True
    }

    zoo2 = {
        label_col: 0.4,
        id_col: 'zoo1',
        'ra': 15.0,
        'dec': -1.0,
        'png_loc': '{}/example_b.png'.format(TEST_EXAMPLE_DIR),
        'fits_loc': '{}/example_b.fits'.format(TEST_EXAMPLE_DIR),
        'png_ready': True
    }

    return pd.DataFrame([zoo1, zoo2] * 128)  # 256 examples