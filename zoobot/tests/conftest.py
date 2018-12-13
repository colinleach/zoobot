import os
import random
import time
import hashlib

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
        'label': 1.
    }



@pytest.fixture()
def parsed_binary_example(parsed_example):
    parsed_example['label'] = 1
    return parsed_example


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


@pytest.fixture()  # depends on create_tfrecord serializing correctly
def serialized_matrix_label_example(size, channels):
    return create_tfrecord.serialize_image_example(
        matrix=np.random.rand(size, size, channels),
        label=1.
        )


@pytest.fixture()  # depends on create_tfrecord serializing correctly
def serialized_matrix_id_example(size, channels, unique_id):
    return create_tfrecord.serialize_image_example(
        matrix=np.random.rand(size, size, channels),
        id_str=unique_id
        )


@pytest.fixture()  # depends on create_tfrecord serializing correctly
def serialized_matrix_label_id_example(size, channels, unique_id):
    return create_tfrecord.serialize_image_example(
        matrix=np.random.rand(size, size, channels),
        label=1.,
        id_str=unique_id
        )


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
def shard_locs(tfrecord_dir, size, channels):  # write shards dynamically when called

    shard_names = [
        's28_shard_0.tfrecord',
        's28_shard_1.tfrecord'
    ]
    tfrecord_locs = list(map(
        lambda x: os.path.join(tfrecord_dir, x),
        shard_names
    ))

    for tfrecord_n, tfrecord_loc in enumerate(tfrecord_locs):
        if os.path.exists(tfrecord_loc):
            os.remove(tfrecord_loc)
        
        examples = [{'matrix': np.random.rand(size, size, channels), 'id_str': str(tfrecord_n) + '_' + str(n)} for n in range(128)]

        writer = tf.python_io.TFRecordWriter(tfrecord_loc)
        for example in examples:  # depends on tfrecord.create_tfrecord
            writer.write(create_tfrecord.serialize_image_example(matrix=example['matrix'], id_str=example['id_str']))
        writer.close()

    return tfrecord_locs  # of form [train_loc, test_loc]




@pytest.fixture()
def tfrecord_matrix_loc(tfrecord_dir, size, channels):  # write shards dynamically when called

    tfrecord_loc = os.path.join(tfrecord_dir, 's28_matrix_0.tfrecord')
    if os.path.exists(tfrecord_loc):
        os.remove(tfrecord_loc)
        
    examples = [{'matrix': np.random.rand(size, size, channels)}]

    writer = tf.python_io.TFRecordWriter(tfrecord_loc)
    for example in examples:  # depends on tfrecord.create_tfrecord
        writer.write(create_tfrecord.serialize_image_example(matrix=example['matrix']))
    writer.close()

    return tfrecord_loc



@pytest.fixture()
def predictor_model_loc():  # not yet on github
    return os.path.join(TEST_EXAMPLE_DIR, 'example_saved_model/1530286779')


@pytest.fixture()
def predictor():
    return lambda x: np.random.rand()  # whatever is passed in, return a single random float score


@pytest.fixture()
def label_col():
    return 'label'


@pytest.fixture()
def id_col():
    return 'id_str'


@pytest.fixture
def columns_to_save(label_col, id_col):
    return [
        label_col,
        id_col,
        'ra',
        'dec']


@pytest.fixture
def unique_id():  # not currently used
    id_hash = hashlib.sha256(bytes(str(time.time()), 'utf8'))  # unique id each time
    yield id_hash.hexdigest()


@pytest.fixture
def catalog(label_col, id_col, unique_id):

    zoo1 = {
        label_col: 1.,
        'ra': 12.0,
        'dec': -1.0,
        'png_loc': '{}/example_a.png'.format(TEST_EXAMPLE_DIR),
        'fits_loc': '{}/example_a.fits'.format(TEST_EXAMPLE_DIR),
        'png_ready': True
    }

    zoo2 = {
        label_col: 0.,
        'ra': 15.0,
        'dec': -1.0,
        'png_loc': '{}/example_b.png'.format(TEST_EXAMPLE_DIR),
        'fits_loc': '{}/example_b.fits'.format(TEST_EXAMPLE_DIR),
        'png_ready': True
    }

    df = pd.DataFrame([zoo1, zoo2] * 128)  # 256 examples
    df[id_col] = [str(n) for n in range(len(df))]
    return df


@pytest.fixture()
def fits_native_dir(tmpdir):
    return tmpdir.mkdir('fits_native').strpath

@pytest.fixture
def catalog_random_images(size, channels, fits_native_dir):
    assert os.path.exists(fits_native_dir)
    n_subjects = 64
    id_strings = [str(n) for n in range(n_subjects)]
    matrices = np.random.rand(n_subjects, size, size, channels)
    relative_fits_locs = ['random_{}.fits'.format(n) for n in range(n_subjects)]
    fits_locs = list(map(lambda rel_loc: os.path.join(fits_native_dir, rel_loc), relative_fits_locs))
    for matrix, loc in zip(matrices, fits_locs):  # write to fits
        hdu = fits.PrimaryHDU(matrix)
        hdu.writeto(loc, overwrite=True)
        assert os.path.isfile(loc)
    catalog = pd.DataFrame(data={'id_str': id_strings, 'fits_loc': fits_locs})
    return catalog


@pytest.fixture()
def db_loc(tmpdir):
    return os.path.join(tmpdir.mkdir('db_dir').strpath, 'db_is_here.db')


@pytest.fixture()
def acquisition_func():
    # Converts loaded subjects to acquisition scores. Here, takes the mean.
    # Must return float, not np.float32, else db will be confused and write as bytes
    def mock_acquisition_callable(matrix_list):
        assert isinstance(matrix_list, list)
        assert all([isinstance(x, np.ndarray) for x in matrix_list])
        assert all([x.shape[0] == x.shape[1] for x in matrix_list])
        return [float(x.mean()) for x in matrix_list]
    return mock_acquisition_callable


@pytest.fixture()
def acquisition():
    return np.random.rand()