import os

import pytest

from tfrecord.create_tfrecord import *


@pytest.fixture()
def example_image_data():
    return np.array(np.ones((50, 50, 3)))


@pytest.fixture
def tfrecord_dir(tmpdir):
    return tmpdir.mkdir('tfrecord_dir').strpath


def test_matrix_to_tfrecord(example_image_data, tfrecord_dir):
    label = 1
    save_loc = '{}/example.tfrecords'.format(tfrecord_dir)
    assert not os.path.exists(save_loc)
    image_to_tfrecord(example_image_data, label, save_loc)
    assert os.path.exists(save_loc)


def test_matrix_to_tfrecord_with_extra_data(example_image_data, tfrecord_dir):
    example_label = 1
    save_loc = '{}/example.tfrecords'.format(tfrecord_dir)
    example_extra_data = {
        'an_int': 1,
        'a_float': .5,
        'some_floats': np.array([1., 2., 3.])
    }
    assert not os.path.exists(save_loc)
    image_to_tfrecord(example_image_data, example_label, save_loc, extra_data=example_extra_data.copy())
    assert os.path.exists(save_loc)

    def read_example(example_loc):
        filename_queue = tf.train.string_input_producer([example_loc],
                                                        num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'matrix': tf.FixedLenFeature([50 ** 2 * 3], tf.float32),
                'an_int': tf.FixedLenFeature([], tf.int64),
                'a_float': tf.FixedLenFeature([], tf.float32),
                'some_floats': tf.FixedLenFeature([3], tf.float32),
            })
        # now return the converted data
        label = features['label']
        image = features['matrix']
        an_int = features['an_int']
        a_float = features['a_float']
        some_floats = features['some_floats']

        return label, image, an_int, a_float, some_floats

    label, image, an_int, a_float, some_floats = read_example(save_loc)  # symbolic examples

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    label, image, an_int, a_float, some_floats = sess.run([label, image, an_int, a_float, some_floats])
    assert label == example_label
    assert image.sum() == example_image_data.sum()
    assert an_int == example_extra_data['an_int']
    assert a_float == example_extra_data['a_float']
    assert all(some_floats == example_extra_data['some_floats'])


def test_matrix_to_tfrecord_with_two_examples(example_image_data, tfrecord_dir):
    save_loc = '{}/example.tfrecords'.format(tfrecord_dir)
    assert not os.path.exists(save_loc)

    example_label_a = 1
    example_extra_data_a = {
        'an_int': 1,
        'a_float': .5,
        'some_floats': np.array([1., 2., 3.])
    }
    example_label_b = 0
    example_extra_data_b = {
        'an_int': 2,
        'a_float': .3,
        'some_floats': np.array([3., 2., 1.])
    }

    image_to_tfrecord(example_image_data, example_label_a, save_loc, extra_data=example_extra_data_a.copy())
    image_to_tfrecord(example_image_data, example_label_b, save_loc, extra_data=example_extra_data_b.copy())
    assert os.path.exists(save_loc)

    def read_example(example_loc):
        filename_queue = tf.train.string_input_producer([example_loc],
                                                        num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'matrix': tf.FixedLenFeature([50 ** 2 * 3], tf.float32),
                'an_int': tf.FixedLenFeature([], tf.int64),
                'a_float': tf.FixedLenFeature([], tf.float32),
                'some_floats': tf.FixedLenFeature([3], tf.float32),
            })
        # now return the converted data
        label = features['label']
        image = features['matrix']
        an_int = features['an_int']
        a_float = features['a_float']
        some_floats = features['some_floats']

        return label, image, an_int, a_float, some_floats

    label, image, an_int, a_float, some_floats = read_example(save_loc)  # symbolic examples

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    label, image, an_int, a_float, some_floats = sess.run([label, image, an_int, a_float, some_floats])
    assert label == example_label_b
    assert image.sum() == example_image_data.sum()
    assert an_int == example_extra_data_b['an_int']
    # assert a_float == example_extra_data_b['a_float']
    assert all(some_floats == example_extra_data_b['some_floats'])

    label, image, an_int, a_float, some_floats = sess.run([label, image, an_int, a_float, some_floats])
    assert label == example_label_a
    assert image.sum() == example_image_data.sum()
    assert an_int == example_extra_data_a['an_int']
    # assert a_float == example_extra_data_a['a_float']
    assert all(some_floats == example_extra_data_a['some_floats'])
