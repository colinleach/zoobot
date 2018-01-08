
import tensorflow as tf
import numpy as np
from functools import partial


def input(filename, mode, size=64, batch=100, copy=True, adjust=True, stratify=False, augment=True):

    dataset = tf.data.TFRecordDataset(filename)
    parse_function = partial(_parse_function, size=size)
    dataset = dataset.map(parse_function)  # Parse the record into tensors.

    if stratify:
        initial_batch = batch * 10
    else:
        initial_batch = batch
    shuffle = initial_batch * 5

    if mode == 'train':
        print('taking a training batch')
        # Repeat the input indefinitely
        # Release in deca-batches to be stratified into batch size
        dataset = dataset.repeat()
        dataset = dataset.shuffle(shuffle).batch(initial_batch)
    elif mode == 'test':
        print('taking an eval batch')
        # Pick one deca-batch of random examples (to be stratified into batch size
        # using .take and then get_next gives only one example
        # using .batch and then get_next gives a batch of examples
        dataset = dataset.shuffle(shuffle).take(initial_batch).batch(initial_batch)

    iterator = dataset.make_one_shot_iterator()
    batch_images, batch_labels = iterator.get_next()

    batch_images = tf.reshape(batch_images, [-1, size, size, 3])
    tf.summary.image('{}/original'.format(mode), batch_images)

    batch_images = tf.reduce_mean(batch_images, axis=3, keepdims=True)
    tf.summary.image('{}/greyscale'.format(mode), batch_images)

    if stratify:
        batch_images = tf.Print(batch_images, [tf.shape(batch_images)], message='images before strat')
        batch_labels = tf.Print(batch_labels, [tf.shape(batch_labels)], message='labels before strat')
        batch_labels = tf.Print(batch_labels, [tf.reduce_sum(batch_labels), tf.reduce_mean(tf.cast(batch_labels, tf.float32))], message='true labels before strat')
        batch_images, batch_labels = stratify_images(batch_images, batch_labels)
        batch_images = tf.Print(batch_images, [tf.shape(batch_images)], message='images after strat')
        batch_labels = tf.Print(batch_labels, [tf.shape(batch_labels)], message='labels after strat')
        batch_labels = tf.Print(batch_labels, [tf.reduce_sum(batch_labels), tf.reduce_mean(tf.cast(batch_labels, tf.float32))], message='true labels after strat')

    if augment:
        batch_images = augment_images(batch_images, copy, adjust)
        tf.summary.image('augmented_{}'.format(mode), batch_images)

    batch_images = tf.Print(batch_images, [tf.shape(batch_images)], message='final images shape')
    batch_labels = tf.Print(batch_labels, [tf.shape(batch_labels)], message='final labels shape')

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


def _parse_function(example_proto, size=64):
    features = {"matrix": tf.FixedLenFeature((size * size * 3), tf.float32),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["matrix"], parsed_features["label"]


#
# def stratify_images_old(batch_images, batch_labels, batch):
#     (batch_images, batch_labels) = tf.contrib.training.stratified_sample(
#         tensors=[batch_images],  # expects a list of tensors, not a single 4d batch tensor
#         labels=batch_labels,
#         target_probs=np.array([0.5, 0.5]),
#         batch_size=batch,
#         enqueue_many=True)
#     batch_images = batch_images[0]
#     return batch_images, batch_labels
