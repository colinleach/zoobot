import os
from collections import Counter

import numpy as np
import pytest
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # don't actually show any figures
import matplotlib.pyplot as plt

from zoobot.estimators import input_utils
from zoobot.tfrecord import read_tfrecord
from zoobot.tests import TEST_EXAMPLE_DIR, TEST_FIGURE_DIR


@pytest.fixture(params=[True, False])
def central(request):
    return request.param

@pytest.fixture()
def batch_of_visual_check_image(visual_check_image):
    return tf.stack([visual_check_image for n in range(16)], axis=0)  # dimensions batch, height, width, channels


"""
Test augmentations applied to a single image (i.e. within map_fn), or batches of images
"""

def test_geometric_augmentations_on_image(visual_check_image, size, central):

    expected_final_size = int(size / 2)
    final_image_tf = input_utils.geometric_augmentation(
        visual_check_image,
        zoom=(1., 1.5),
        final_size=expected_final_size,
        central=central
    )

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        original_image = session.run(visual_check_image)
        final_image = np.squeeze(session.run(final_image_tf))  # remove batch dimension
        assert final_image.shape == (expected_final_size, expected_final_size, 3)

        fig, axes = plt.subplots(ncols=2)
        axes[0].imshow(original_image)
        axes[0].set_title('Before')
        axes[1].imshow(final_image)
        axes[1].set_title('After')
        fig.tight_layout()
        fig.savefig(os.path.join(TEST_FIGURE_DIR, 'geometric_augmentation_check_single_image.png'))


def test_photometric_augmentations_on_image(visual_check_image):
    final_image = input_utils.photographic_augmentation(visual_check_image, max_brightness_delta=0.1, contrast_range=(0.9, 1.1))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        input_image = session.run(visual_check_image)
        final_image = np.squeeze(session.run(final_image))  # adds a batch dimension

    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(input_image)
    axes[0].set_title('Before')
    print(final_image.shape)  # TODO assert
    axes[1].imshow(final_image)
    axes[1].set_title('After')
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'photometric_augmentation_check_single_image.png'))



def test_repeated_geometric_augmentations_on_image(batch_of_visual_check_image, central):
    transformed_images = input_utils.geometric_augmentation(
        batch_of_visual_check_image,
        zoom=(1., 1.5),
        final_size=256,
        central=central
    )

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        transformed_images = session.run(transformed_images)

    fig, axes = plt.subplots(nrows=16, figsize=(4, 4 * 16))
    for image_n, image in enumerate(transformed_images):
        axes[image_n].imshow(image)
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'geometric_augmentation_check_on_batch.png'))


def test_repeated_photometric_augmentations_on_image(batch_of_visual_check_image):
    transformed_images = input_utils.photographic_augmentation(batch_of_visual_check_image, max_brightness_delta=0.1, contrast_range=(0.9, 1.1))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        transformed_images = session.run(transformed_images)

    fig, axes = plt.subplots(nrows=16, figsize=(4, 4 * 16))
    for image_n, image in enumerate(transformed_images):
        axes[image_n].imshow(image)
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'photometric_augmentation_check_on_batch.png'))


def test_all_augmentations_on_batch(batch_of_visual_check_image):

    input_config = input_utils.InputConfig(
        name='pytest',
        tfrecord_loc='',
        label_cols=[''],
        initial_size=424,
        final_size=256,
        channels=3,
        batch_size=16,
        stratify=False,
        regression=False,
        repeat=False,
        shuffle=False,
        geometric_augmentation=True,
        shift_range=None,
        zoom=(1., 1.5),
        fill_mode=None,
        photographic_augmentation=True,
        max_brightness_delta=0.2,
        contrast_range=(0.8, 1.2)
    )

    transformed_batch = input_utils.augment_images(batch_of_visual_check_image, input_config)

    with tf.Session() as sess:
        transformed_batch = sess.run(transformed_batch)

    assert not isinstance(transformed_batch, list)  # should be a single 4D tensor, not a list
    transformed_images = [transformed_batch[n] for n in range(len(transformed_batch))]  # back to list form
    fig, axes = plt.subplots(nrows=len(transformed_images), figsize=(4, 4 * len(transformed_images)))
    for image_n, image in enumerate(transformed_images):
        axes[image_n].imshow(image)
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'all_augmentations_check.png'))


"""
Tests for label-related loading
"""

@pytest.fixture
def label_cols():
    return ['label_a', 'label_b']


@pytest.fixture
def labels(label_cols, n_examples):
    label_values = [np.random.rand(n_examples) for _ in label_cols]
    labels = dict(zip(label_cols, label_values))
    return labels

# TODO surely duplicated?
@pytest.fixture
def id_strs(n_examples):
    return [n for n in range(n_examples)]

@pytest.fixture
def multi_label_dataset(random_features, labels, id_strs, batch_size):
    images = random_features['x']
    data = {'matrix': images, 'id_strs': id_strs}
    data.update(labels)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size)
    return dataset


@pytest.fixture
def multi_label_batch(multi_label_dataset):
    iterator = multi_label_dataset.make_one_shot_iterator()
    return iterator.get_next()


def test_get_labels_from_batch(multi_label_batch, label_cols, batch_size):
    labels_tf = input_utils.get_labels_from_batch(multi_label_batch, label_cols)
    with tf.Session() as sess:  
        labels = sess.run(labels_tf)
    assert labels.shape == (batch_size, len(label_cols))


@pytest.fixture(params=[
    {'matrix': 'string', 'id_str': 'string'}, 
    {'matrix': 'string', 'id_str': 'string', 'label_a': 'float', 'label_b': 'float'}
])
def requested_features(request):
    return request.param

@pytest.fixture
def tfrecord_loc(requested_features, tfrecord_multilabel_loc, tfrecord_matrix_id_loc):
    if 'label_a' in requested_features.keys():
        return tfrecord_multilabel_loc
    else:
        return tfrecord_matrix_id_loc

def test_get_batch(requested_features, tfrecord_loc, size, channels):
    feature_spec = read_tfrecord.get_feature_spec(requested_features)  # not actually under test, a bit sloppy
    batch = input_utils.get_batch(
        [tfrecord_loc],
        feature_spec,
        batch_size=24,  # larger batch size to make sure we loop, otherwise only 100 elements
        shuffle=True,
        repeat=True
    )
    with tf.Session() as sess:  
        batches = []
        for _ in range(10):
            batches.append(sess.run(batch))
    assert len(batches) == 10

    id_strs = [id_str.decode('utf-8') for b in batches for id_str in b['id_str']]
    assert '12' in set(id_strs)
    assert len(set(id_strs)) == 128  # all ids in tfrecord



def test_get_batch_double_locs(tfrecord_matrix_id_loc, tfrecord_matrix_id_loc_distinct):
    feature_spec = read_tfrecord.id_feature_spec()
    tfrecord_locs = [tfrecord_matrix_id_loc, tfrecord_matrix_id_loc_distinct]

    n_batches = 81  # enough to cycle through full records, and get close to expected counts
    shuffle = False

    # load individually
    batch_tf_0 = input_utils.get_batch(tfrecord_locs[0], feature_spec, batch_size=24, shuffle=shuffle, repeat=True)
    with tf.Session() as sess:  
        batches_tf_0 = []
        for i in range(n_batches):
            batches_tf_0.append(sess.run(batch_tf_0))
    assert len(batches_tf_0) == n_batches

    batch_tf_1 = input_utils.get_batch(tfrecord_locs[1], feature_spec, batch_size=24, shuffle=shuffle, repeat=True)
    with tf.Session() as sess:  
        batches_tf_1 = []
        for i in range(n_batches):
            batches_tf_1.append(sess.run(batch_tf_1))
    assert len(batches_tf_1) == n_batches

    # load from both
    batch = input_utils.get_batch(tfrecord_locs, feature_spec, batch_size=24, shuffle=shuffle, repeat=True)
    with tf.Session() as sess:  
        batches = []
        for i in range(n_batches):
            batches.append(sess.run(batch))
    assert len(batches) == n_batches

    id_strs = [id_str.decode('utf-8') for b in batches for id_str in b['id_str']]
    id_strs_0 = [id_str.decode('utf-8') for b in batches_tf_0 for id_str in b['id_str']]
    id_strs_1 = [id_str.decode('utf-8') for b in batches_tf_1 for id_str in b['id_str']]
    # for batch in batches:
        # print(batch)
    # assert False

    #  for tests to work, should be some ids only in one record or the other
    assert len(set(id_strs_0) ^ set(id_strs_1)) > 0

    #  should have loaded all ids in double-mode (n_batches is plenty to cycle through both)
    assert set(id_strs_0) | set(id_strs_1) == set(id_strs)

    counts_0 = Counter(id_strs_0)
    counts_1 = Counter(id_strs_1)
    counts = Counter(id_strs)

    # check that for all tfrecords, looped through all elements evenly (3-4 times for double, 7-8 times for single, as records are shorter)
    for counter in [counts_0, counts_1, counts]:
        most_common_n_reads = counter.most_common()[0][1]
        least_common_n_reads = counter.most_common()[::-1][0][1]
        assert most_common_n_reads == least_common_n_reads + 1  # batch size won't exactly match record size

    mean_tfrecord_0_reads = np.mean([counts[id_str] for id_str in id_strs_0])
    mean_tfrecord_1_reads = np.mean([counts[id_str] for id_str in id_strs_1])
    # images should be read at approximately the same rate, regardless of the size of the tfrecord that holds them
    assert np.abs(mean_tfrecord_0_reads - mean_tfrecord_1_reads) < 0.2



def test_predict_input_func_subbatch_with_labels(tfrecord_matrix_ints_loc, size):
    
    # tfrecord_matrix_loc
    n_galaxies = 24
    subjects, labels, _ = input_utils.predict_input_func(
        tfrecord_matrix_ints_loc,
        n_galaxies=n_galaxies,
        initial_size=size,
        mode='labels'
    )
    with tf.Session() as sess:
        subjects = sess.run(subjects)
        assert subjects.shape == (n_galaxies, size, size, 3)
        labels = sess.run(labels)
        assert len(labels) == n_galaxies
        # should not have shuffled
        assert labels[0] < labels [1] < labels [2] < labels [10] < labels[23]

def test_predict_input_func_with_id(shard_locs, size):
    n_galaxies = 24
    tfrecord_loc = shard_locs[0]
    subjects, _, id_strs = input_utils.predict_input_func(tfrecord_loc, n_galaxies=n_galaxies, initial_size=size, mode='id_str')
    with tf.Session() as sess:
        subjects, id_strs = sess.run([subjects, id_strs])
    assert subjects.shape == (n_galaxies, size, size, 3)  # does not do augmentations, that happens at predict time
    assert len(id_strs) == 24


def test_predict_input_func_subbatch_no_labels(tfrecord_matrix_loc, size):
    n_galaxies = 24
    subjects, _, _ = input_utils.predict_input_func(tfrecord_matrix_loc, n_galaxies=n_galaxies, initial_size=size, mode='matrix')
    with tf.Session() as sess:
        subjects = sess.run(subjects)
    assert subjects.shape == (n_galaxies, size, size, 3)  # does not do augmentations, that happens at predict time
