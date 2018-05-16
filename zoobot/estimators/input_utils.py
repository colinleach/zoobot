import numpy as np
import tensorflow as tf

from zoobot.tfrecord.tfrecord_io import load_dataset


def input(tfrecord_loc, name, size=64, batch_size=100, stratify=False, transform=True, adjust=True):
    """
    Load tfrecord as dataset. Stratify and augment images as directed. Batch with queues for Estimator input.
    Args:
        tfrecord_loc (str): file location of tfrecord to load
        name (str): name (e.g. 'train' for Tensorboard records
        size (int): length of square dimension of images e.g. 128
        batch_size (int): size of batch to return
        stratify (bool): split batch into even True/False examples. Not currently very accurate for small batches!
        transform (bool): augment images with flips, 90' rotations
        adjust (bool): augment images with random brightness/contrast

    Returns:
        (dict) of form {'x': greyscale image batch}, represented as np.float32 Tensor of shape [batch, size, size, 1]}
        (Tensor) categorical labels for each image
    """

    feature_spec = matrix_label_feature_spec(size)
    dataset = load_dataset(tfrecord_loc, feature_spec)
    dataset = dataset.shuffle(batch_size * 10)

    # queue single images into batches
    if stratify:
        iterator = dataset.make_one_shot_iterator()
        example = iterator.get_next()
        batch_image = example['matrix']
        batch_label = example['label']
        batch_images, batch_labels = stratify_images(batch_image, batch_label, batch_size)
    else:
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        batch_images, batch_labels = batch['matrix'], batch['label']  # smart enough to handle dict calls to batches!
        # batch_images, batch_labels = tf.train.batch(
        #     [batch_image, batch_label],
        #     batch,
        #     capacity=batch * 100)

    batch_images = tf.reshape(batch_images, [-1, size, size, 3])
    tf.summary.image('{}/original'.format(name), batch_images)

    batch_images = tf.reduce_mean(batch_images, axis=3, keepdims=True)
    tf.summary.image('{}/greyscale'.format(name), batch_images)

    batch_images = augment_images(batch_images, transform, adjust)
    tf.summary.image('augmented_{}'.format(name), batch_images)

    feature_cols = {'x': batch_images}
    return feature_cols, batch_labels


def stratify_images(image, label, batch_size):
    """
    Queue examples of images/labels into roughly even True/False counts
    Args:
        image (Tensor): pixel values for one galaxy. Should be from queue e.g. iterator.get_next() to batch properly.
        label (Tensor): label for one galaxy. Should be from queue e.g. iterator.get_next() to batch properly.
        batch_size (int): size of batch to return

    Returns:
        (Tensor): pixel value batch of 1st dim length batch_size, with other dimensions set by image dimensions
        (Tensor): label batch of 1st dim length batch_size
    """
    data_batch, label_batch = tf.contrib.training.stratified_sample(
        [image],
        label,
        target_probs=np.array([0.5, 0.5]),
        batch_size=batch_size,
        enqueue_many=False,  # each image/label is a single example, will be automatically batched (thanks TensorFlow!)
        queue_capacity=batch_size * 10
    )
    return data_batch, label_batch


def augment_images(images, transform=True, adjust=True):
    if transform:
        images = tf.map_fn(transform_3d, images)
    if adjust:
        images = tf.image.random_brightness(images, max_delta=0.1)
        images = tf.image.random_contrast(images, lower=0.9, upper=1.1)  # TODO not sure these look great, should test
    return images


def transform_3d(images):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.rot90(images)
    return images


def matrix_label_parse_function(example_proto, size=64):
    features = matrix_label_feature_spec(size)
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["matrix"], parsed_features["label"]


def matrix_label_feature_spec(size):
    return {
        "matrix": tf.FixedLenFeature((size * size * 3), tf.float32),
        "label": tf.FixedLenFeature((), tf.int64)}
