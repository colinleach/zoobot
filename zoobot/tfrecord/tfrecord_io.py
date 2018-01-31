import os
from functools import partial

import numpy as np
import tensorflow as tf

from zoobot.tfrecord.create_tfrecord import image_to_tfrecord


def read_first_example(example_loc, feature_spec):
    dataset = load_dataset(example_loc, feature_spec)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def load_dataset(example_loc, feature_spec):
    dataset = tf.data.TFRecordDataset(example_loc)
    parse_function = partial(tf.parse_single_example, features=feature_spec)
    return dataset.map(parse_function)  # Parse the record into tensors


def matrix_label_feature_spec(size):
    return {
        "matrix": tf.FixedLenFeature((size * size * 3), tf.float32),
        "label": tf.FixedLenFeature((), tf.int64)}


if __name__ == '__main__':

    example_image_data = np.array(np.ones((50, 50, 3), dtype=float))
    tfrecord_dir = '/data/repos/zoobot/zoobot/catalogs/tfrecord'
    label = 1
    save_loc = '{}/example.tfrecords'.format(tfrecord_dir)
    if os.path.exists(save_loc):
        os.remove(save_loc)
    assert not os.path.exists(save_loc)
    writer = tf.python_io.TFRecordWriter(save_loc)
    image_to_tfrecord(example_image_data, label, writer)
    assert os.path.exists(save_loc)
    writer.close()  # very important - will give 'DataLoss' error if writer not closed

    feature_spec = matrix_label_feature_spec(size=50)
    example_features = read_first_example(save_loc, feature_spec)
    label = example_features['label']
    image = example_features['matrix']

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    label, image = sess.run([label, image])

    print(label, image)
