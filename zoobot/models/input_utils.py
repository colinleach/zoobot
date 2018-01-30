
import tensorflow as tf
import numpy as np
from functools import partial


def input(filename, mode, size=64, batch=100, copy=True, adjust=True, stratify=False, augment=True):

    dataset = tf.data.TFRecordDataset(filename)
    parse_function = partial(matrix_label_parse_function, size=size)
    dataset = dataset.map(parse_function)  # Parse the record into tensors.

    if mode == 'train':
        print('taking a training batch')
        # Repeat the input indefinitely
        # Release in deca-batches to be stratified into batch size
        dataset = dataset.shuffle(batch * 10)
    elif mode == 'test':
        print('taking an eval batch')
        # Restrict to one deca-batch of random examples (to avoid looping forever)
        # using .take and then get_next gives only one example (good)
        # using .batch and then get_next gives a batch of examples (too early)
        dataset = dataset.shuffle(batch * 10)

    iterator = dataset.make_one_shot_iterator()
    batch_image, batch_label = iterator.get_next()

    # queue single images into batches
    if stratify:
        batch_images, batch_labels = stratify_images_auto(batch_image, batch_label, batch)
    else:
        batch_images, batch_labels = tf.train.batch(
            [batch_image, batch_label],
            batch,
            capacity=batch)

    batch_images = tf.reshape(batch_images, [-1, size, size, 3])
    tf.summary.image('{}/original'.format(mode), batch_images)

    batch_images = tf.reduce_mean(batch_images, axis=3, keepdims=True)
    tf.summary.image('{}/greyscale'.format(mode), batch_images)

    if augment:
        batch_images = augment_images(batch_images, copy, adjust)
        tf.summary.image('augmented_{}'.format(mode), batch_images)

    feature_cols = {'x': batch_images}
    return feature_cols, batch_labels


def stratify_images(batch_images, batch_labels):

    true_images_filter = tf.cast(batch_labels, tf.bool)
    print('batch labels', batch_labels.shape)
    print('batch images', batch_images.shape)
    true_images = tf.boolean_mask(batch_images, true_images_filter)
    print('true images', true_images.shape)

    false_images_filter = tf.cast(tf.abs(batch_labels - tf.ones_like(batch_labels)), tf.bool)

    false_images = tf.boolean_mask(batch_images, false_images_filter)

    # Filter!
    n_true_examples = tf.to_int32(tf.reduce_sum(batch_labels))
    n_false_examples = tf.to_int32(tf.size(batch_labels) - n_true_examples)

    # if there are fewer true examples, we should return true examples * 2
    # otherwise, vica versa: return false examples * 2
    condition = tf.less(x=n_true_examples, y=n_false_examples)
    n_examples = tf.cond(
        condition,
        true_fn=lambda: n_true_examples,
        false_fn=lambda: n_false_examples
    )

    true_images = true_images[:n_examples, :, :, :]
    false_images = false_images[:n_examples, :, :, :]

    true_labels = tf.ones([n_examples], dtype=tf.int32)
    false_labels = tf.zeros([n_examples], dtype=tf.int32)

    batch_images = tf.concat([true_images, false_images], axis=0)
    batch_labels = tf.concat([true_labels, false_labels], axis=0)

    return batch_images, batch_labels


def stratify_images_auto(batch_images, batch_labels, batch_size):
    data_batch, label_batch = tf.contrib.training.stratified_sample(
        [batch_images],  # list of tensors. enqueue many ensures treated with batch dimension within list
        batch_labels,
        target_probs=np.array([0.5, 0.5]),
        batch_size=batch_size,
        enqueue_many=False,
        queue_capacity=batch_size * 10
    )
    return data_batch, label_batch


def rejection_sample(batch_images, batch_labels, batch_size):
    data_batch, label_batch = tf.contrib.training.rejection_sample(
        [batch_images, batch_labels],
        accept_prob_fn=lambda x: 0.5,  # for now
        batch_size=2,
        enqueue_many=True,
    )
    return data_batch, label_batch


def augment_images(images, copy=True, adjust=True):
    if copy:
        images = tf.map_fn(transform_3d, images)
    if adjust:
        images = tf.image.random_brightness(images, max_delta=0.1)
        images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
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
