import os

import pytest
from zoobot.tfrecord.create_tfrecord import *

from zoobot.tfrecord.tfrecord_io import matrix_label_feature_spec, read_first_example, load_dataset


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
    writer = tf.python_io.TFRecordWriter(save_loc)
    image_to_tfrecord(example_image_data, label, writer)
    assert os.path.exists(save_loc)
    writer.close()  # important

    # will use convenience parser/spec from tfrecord.io
    feature_spec = matrix_label_feature_spec(size=50)
    # load tfrecord as dataset, read the first example with parser/spec
    example_features = read_first_example(save_loc, feature_spec)
    saved_label = example_features['label']
    saved_image = example_features['matrix']
    # execute graph (with queuerunners)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        saved_label, saved_image = sess.run([saved_label, saved_image])
        assert saved_label == label
        # image returned flat, float32 dtype
        assert saved_image == pytest.approx(example_image_data.flatten().astype(np.float32))


def extra_data_feature_spec():
    return {
        'label': tf.FixedLenFeature([], tf.int64),
        'matrix': tf.FixedLenFeature([50 ** 2 * 3], tf.float32),
        'an_int': tf.FixedLenFeature([], tf.int64),
        'a_float': tf.FixedLenFeature([], tf.float32),
        'some_floats': tf.FixedLenFeature([3], tf.float32)}


def extra_data_parse_function(example_proto, feature_spec=extra_data_feature_spec()):
    parsed_features = tf.parse_single_example(example_proto, feature_spec)
    return parsed_features["matrix"], parsed_features["label"], parsed_features["an_int"], parsed_features["a_float"], \
           parsed_features["some_floats"]


def test_matrix_to_tfrecord_with_extra_data(example_image_data, tfrecord_dir):
    example_label = 1
    save_loc = '{}/example.tfrecords'.format(tfrecord_dir)
    example_extra_data = {
        'an_int': 1,
        'a_float': .5,
        'some_floats': np.array([1., 2., 3.])
    }
    assert not os.path.exists(save_loc)
    writer = tf.python_io.TFRecordWriter(save_loc)
    image_to_tfrecord(example_image_data, example_label, writer, extra_data=example_extra_data.copy())
    assert os.path.exists(save_loc)
    writer.close()

    example_features = read_first_example(save_loc, extra_data_feature_spec())

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        example_features = sess.run(example_features)
        label = example_features['label']
        image = example_features['matrix']
        an_int = example_features['an_int']
        a_float = example_features['a_float']
        some_floats = example_features['some_floats']

        assert label == example_label
        assert image == pytest.approx(example_image_data.flatten().astype(np.float32))
        assert an_int == example_extra_data['an_int']
        assert a_float == pytest.approx(example_extra_data['a_float'])
        assert some_floats == pytest.approx(example_extra_data['some_floats'].flatten().astype(np.float32))


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

    writer = tf.python_io.TFRecordWriter(save_loc)
    image_to_tfrecord(example_image_data, example_label_a, writer, extra_data=example_extra_data_a.copy())
    image_to_tfrecord(example_image_data, example_label_b, writer, extra_data=example_extra_data_b.copy())
    assert os.path.exists(save_loc)
    writer.close()

    dataset = load_dataset(save_loc, extra_data_feature_spec())
    iterator = dataset.make_one_shot_iterator()

    example_a = iterator.get_next()
    example_b = iterator.get_next()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        example_a = sess.run(example_a)
        assert example_a['label'] == example_label_a
        assert example_a['matrix'] == pytest.approx(example_image_data.flatten().astype(np.float32))
        assert example_a['an_int'] == example_extra_data_a['an_int']
        assert example_a['a_float'] == pytest.approx(example_extra_data_a['a_float'])
        assert example_a['some_floats'] == pytest.approx(example_extra_data_a['some_floats'])

        example_b = sess.run(example_b)
        assert example_b['label'] == example_label_b
        assert example_b['matrix'] == pytest.approx(example_image_data.flatten().astype(np.float32))
        assert example_b['an_int'] == example_extra_data_b['an_int']
        assert example_b['a_float'] == pytest.approx(example_extra_data_b['a_float'])
        assert example_b['some_floats'] == pytest.approx(example_extra_data_b['some_floats'])
