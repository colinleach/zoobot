import os
import random

import numpy as np
import pytest
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from zoobot.estimators import input_utils
from zoobot.tfrecord import create_tfrecord

TEST_EXAMPLE_DIR = 'zoobot/test_examples'


"""
Test augmentation applied to a single image (i.e. within map_fn)
"""

@pytest.fixture()
# actual image used for visual checks
def visual_check_image():
    return np.array(Image.open('zoobot/test_examples/example_b.png'))


def test_augmentations_on_image(visual_check_image):
    result = input_utils.transform_3d(visual_check_image)

    with tf.Session() as sess:
        result = sess.run(result)

        fig, axes = plt.subplots(ncols=2)
        axes[0].imshow(visual_check_image)
        axes[0].set_title('Before')
        axes[1].imshow(result)
        axes[1].set_title('After')
        fig.tight_layout()
        fig.savefig('zoobot/test_examples/augmentation_check_single_image.png')
        fig.show()  # should always appear rotated, 75% chance of flipped


def test_repeated_augmentations_on_image(visual_check_image):
    transformed_images = [input_utils.transform_3d(visual_check_image) for n in range(6)]

    with tf.Session() as sess:
        transformed_images = sess.run(transformed_images)

        fig, axes = plt.subplots(nrows=6, figsize=(4, 4 * 6))
        for image_n, image in enumerate(transformed_images):
            axes[image_n].imshow(image)
        fig.tight_layout()
        fig.savefig('zoobot/test_examples/augmentation_check_on_batch.png')
        fig.show()


"""
Test augmentation applied by map_fn to a chain of images from from_tensor_slices
"""



@pytest.fixture()
def batch_of_visual_check_image(visual_check_image):
    return np.array([visual_check_image for n in range(16)])  # dimensions batch, height, width, channels
#
#
# def test_augmentations_on_batch(batch_of_visual_check_image):
#     transformed_batch = input_utils.transform_3d(batch_of_visual_check_image)
#     transformed_images = [transformed_batch[n] for n in range(len(transformed_batch))]  # back to list form
#     fig, axes = plt.subplots(ncols=len(transformed_images), figsize=(4, 4 * len(transformed_images)))
#     for image_n, image in enumerate(transformed_images):
#         axes[image_n].imshow(image)
#     fig.tight_layout()
#     fig.save('zoobot/test_examples/augmentation_check.png')
#     fig.show()


@pytest.fixture()
def benchmark_image():
    single_channel = np.array([[1., 2., 3., 4.] for n in range(4)])  # each channel has rows of 1 2 3 4
    return np.array([single_channel for n in range(3)])  # copied 3 times




@pytest.fixture(scope='module')
def size():
    return 4


@pytest.fixture(scope='module')
def true_image_values():
    return 3.


@pytest.fixture(scope='module')
def false_image_values():
    return -3.


@pytest.fixture()
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



"""
Functional test on fake data, saved to temporary tfrecords
"""


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

#
# def test_input_utils(tfrecord_dir, example_tfrecords, size, true_image_values, false_image_values):
#
#     # example_tfrecords sets up the tfrecords to read - needs to be an arg but is implicitly called by pytest
#
#     train_batch = 64
#     test_batch = 128
#
#     train_loc = tfrecord_dir + '/train.tfrecords'
#     test_loc = tfrecord_dir + '/test.tfrecords'
#     assert os.path.exists(train_loc)
#     assert os.path.exists(test_loc)
#
#     train_config = input_utils.InputConfig(
#         name='train',
#         tfrecord_loc=train_loc,
#         image_dim=size,
#         channels=3,
#         label_col=None,  # TODO not sure about this
#         batch_size=train_batch,
#         stratify=False,
#         stratify_probs=None,
#         transform=False
#     )
#     train_features, train_labels = input_utils.get_input(train_config)
#     train_images = train_features['x']
#
#     train_strat_config = input_utils.InputConfig(
#         name='train',
#         tfrecord_loc=train_loc,
#         image_dim=size,
#         channels=3,
#         label_col=None,  # TODO not sure about this
#         batch_size=train_batch,
#         stratify=True,
#         stratify_probs=np.array([0.8, 0.2]),
#         transform=False
#     )
#     train_features_strat, train_labels_strat = input_utils.get_input(train_strat_config)
#     train_images_strat = train_features_strat['x']
#
#     test_config = input_utils.InputConfig(
#         name='test',
#         tfrecord_loc=test_loc,
#         image_dim=size,
#         channels=3,
#         label_col=None,  # TODO not sure about this
#         batch_size=test_batch,
#         stratify=False,
#         stratify_probs=None,
#         transform=False
#     )
#     test_features, test_labels = input_utils.get_input(test_config)
#     test_images = test_features['x']
#
#     test_strat_config = input_utils.InputConfig(
#         name='test_strat',
#         tfrecord_loc=test_loc,
#         image_dim=size,
#         channels=3,
#         label_col=None,  # TODO not sure about this
#         batch_size=test_batch,
#         stratify=True,
#         stratify_probs=np.array([0.8, 0.2]),
#         transform=False
#     )
#     test_features_strat, test_labels_strat = input_utils.get_input(test_strat_config)
#     test_images_strat = test_features_strat['x']
#
#     with tf.train.MonitoredSession() as sess:  # mimic Estimator environment
#
#         train_images, train_labels = sess.run([train_images, train_labels])
#         assert len(train_labels) == train_batch
#         assert train_images.shape[0] == train_batch
#         assert train_labels.mean() < .6  # should not be stratified
#         assert train_images.shape == (train_batch, size, size, 1)
#         verify_images_match_labels(train_images, train_labels, true_image_values, false_image_values, size)
#
#         train_images_strat, train_labels_strat = sess.run([train_images_strat, train_labels_strat])
#         assert len(train_labels_strat) == train_batch
#         assert train_images_strat.shape[0] == train_batch
#         assert train_labels_strat.mean() < 0.75 and train_labels_strat.mean() > 0.25  # stratify not very accurate...
#         assert train_images_strat.shape == (train_batch, size, size, 1)
#         verify_images_match_labels(train_images_strat, train_labels_strat, true_image_values, false_image_values, size)
#
#         test_images, test_labels = sess.run([test_images, test_labels])
#         assert len(test_labels) == test_batch
#         assert test_images.shape[0] == test_batch
#         assert test_labels.mean() < 0.6  # should not be stratified
#         assert test_images.shape == (test_batch, size, size, 1)
#         verify_images_match_labels(test_images, test_labels, true_image_values, false_image_values, size)
#
#         test_images_strat, test_labels_strat = sess.run([test_images_strat, test_labels_strat])
#         assert len(test_labels_strat) == test_batch
#         assert test_images_strat.shape[0] == test_batch
#         assert test_labels_strat.mean() < 0.75 and test_labels_strat.mean() > 0.25  # stratify not very accurate...
#         assert test_images_strat.shape == (test_batch, size, size, 1)
#         verify_images_match_labels(test_images_strat, test_labels_strat, true_image_values, false_image_values, size)
#
#
# def verify_images_match_labels(images, labels, true_values, false_values, size):
#     for example_n in range(len(labels)):
#         if labels[example_n] == 1:
#             expected_values = true_values
#         else:
#             expected_values = false_values
#         expected_matrix = np.ones((size, size, 1), dtype=np.float32) * expected_values
#         assert images[example_n, :, :, :] == pytest.approx(expected_matrix)
