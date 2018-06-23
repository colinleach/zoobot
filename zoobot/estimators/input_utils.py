import functools

import numpy as np
import tensorflow as tf
from scipy import ndimage

from zoobot.tfrecord.tfrecord_io import load_dataset


def input(tfrecord_loc, input_params):
    """
    Load tfrecord as dataset. Stratify and transform_3d images as directed. Batch with queues for Estimator input.
    Args:
        tfrecord_loc (str): file location of tfrecord to load
        name (str): name (e.g. 'train' for Tensorboard records
        size (int): length of square dimension of images e.g. 128
        batch_size (int): size of batch to return
        stratify (bool): split batch into even True/False examples. Not currently very accurate for small batches!
        transform (bool): transform_3d images with flips, 90' rotations
        adjust (bool): transform_3d images with random brightness/contrast

    Returns:
        (dict) of form {'x': greyscale image batch}, represented as np.float32 Tensor of shape [batch, size, size, 1]}
        (Tensor) categorical labels for each image
    """
    with tf.name_scope('input_{}'.format(input_params['name'])):
        feature_spec = matrix_label_feature_spec(input_params['image_dim'], input_params['channels'])
        dataset = load_dataset(tfrecord_loc, feature_spec)
        dataset = dataset.shuffle(input_params['batch_size'] * 10)

        dataset = dataset.batch(input_params['batch_size'])
        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        batch_images, batch_labels = batch['matrix'], batch['label']

        if input_params['stratify']:
            #  warning - stratify only works if initial probabilities are specified
            batch_images_stratified, batch_labels_stratified = stratify_images(
                batch_images,
                batch_labels,
                input_params['batch_size'],
                input_params['stratify_probs']
            )

            batch_images = batch_images_stratified
            batch_labels = batch_labels_stratified

        preprocessed_batch_images = preprocess_batch(batch_images, input_params)

        return preprocessed_batch_images, batch_labels


def preprocess_batch(batch_images, input_params):
    with tf.name_scope('preprocess'):
        batch_images = tf.reshape(
            batch_images,
            [-1, input_params['image_dim'], input_params['image_dim'],
             input_params['channels']])
        tf.summary.image('preprocess/original', batch_images)

        batch_images = tf.reduce_mean(batch_images, axis=3, keepdims=True)
        tf.summary.image('preprocess/greyscale', batch_images)

        batch_images = augment_images(batch_images, input_params)
        tf.summary.image('augmented', batch_images)

        feature_cols = {'x': batch_images}
        assert feature_cols is not None
        return feature_cols


def stratify_images(image, label, batch_size, init_probs):
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

    assert init_probs is not None  # should not be called with stratify=False
    data_batch, label_batch = tf.contrib.training.stratified_sample(
        [image],
        label,
        target_probs=np.array([0.5, 0.5]),
        init_probs=init_probs,
        batch_size=batch_size,
        enqueue_many=True,  # each image/label is a single example, will be automatically batched (thanks TensorFlow!)
        queue_capacity=batch_size * 100
    )
    return data_batch, label_batch


def augment_images(images, input_params):
    if input_params['transform']:
        images = tf.map_fn(transform_3d, images)
    return images
#
# def crop_to_random_boxes(images, input_params):
#
#     # Generate a single distorted bounding box.
#     begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
#         image_size=tf.constant([input_params['image_dim'], input_params['image_dim'], input_params['channels']], dtype=tf.int32),
#         bounding_boxes=tf.ones((len(images), 1, 4), tf.int32) * tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32),
#         min_object_covered=0.95)
#
#     # Draw the bounding box in an image summary.
#     image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
#                                                   bbox_for_draw)
#     tf.summary.image('images_with_box', image_with_box)
#
#     # Employ the bounding box to distort the image.
#     distorted_image = tf.slice(image, begin, size
#     return images
# TODO
# https://www.tensorflow.org/api_docs/python/tf/image/sample_distorted_bounding_box
# https://www.tensorflow.org/api_docs/python/tf/image/crop_to_bounding_box

# def augment_images(images, params):
#     if params['transform']:
#         # images = tf.map_fn(
#         #     lambda image: tf.py_func(
#         #         func=functools.partial(transform_3d, params=params),
#         #         inp=[image],
#         #         Tout=tf.float32,
#         #         stateful=False,
#         #         name='augment'
#         #     ),
#         #     images)
#
#         [images] = tf.py_func(
#                 func=functools.partial(transform_3d, params=params),
#                 inp=[images[n] for n in range(images.shape[0])],
#                 Tout=tf.float32,
#                 stateful=False,
#                 name='augment')
#         images = tf.concat(images, axis=3)
#
#         images = tf.image.random_flip_left_right(images)
#         images = tf.image.random_flip_up_down(images)
#
#     # TODO These don't look good in practice. Perhaps standardisation needed?
#     # if params['adjust']:
#     #     images = tf.image.random_brightness(images, max_delta=0.1)
#     #     images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
#     return images


def matrix_label_feature_spec(size, channels):
    return {
        "matrix": tf.FixedLenFeature((size * size * channels), tf.float32),
        "label": tf.FixedLenFeature((), tf.int64)}


def matrix_feature_spec(size, channels):  # used for predict mode
    return {
        "matrix": tf.FixedLenFeature((size * size * channels), tf.float32)}


def transform_3d(images):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.rot90(images)
    # TODO zoom and off-center!
    return images


#
# def transform_3d(x, params):
#     # x is a single image, so it doesn't have image number at index 0
#     # slightly tailored from Keras source code (Francois, Keras author)
#     img_row_index = 0
#     img_col_index = 1
#     img_channel_index = 2
#
#     # find transform matrices (numpy)
#
#     # use composition of homographies to generate final transform that needs to be applied
#     if params['rotation_range']:
#         theta = np.pi / 180 * np.random.uniform(-params['rotation_range'], params['rotation_range'])
#     else:
#         theta = 0
#     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
#                                 [np.sin(theta), np.cos(theta), 0],
#                                 [0, 0, 1]])
#     if params['height_shift_range']:
#         tx = np.random.uniform(
#             -params['height_shift_range'],
#             params['height_shift_range']) * int(x.shape[img_row_index])  # shape type 'Dimension'
#     else:
#         tx = 0
#
#     if params['width_shift_range']:
#         ty = np.random.uniform(
#             -params['width_shift_range'],
#             params['width_shift_range'] * int(x.shape[img_col_index]))  # shape type 'Dimension'
#     else:
#         ty = 0
#
#     translation_matrix = np.array([[1, 0, tx],
#                                    [0, 1, ty],
#                                    [0, 0, 1]])
#
#     # TODO disable shear option for now
#     # if params['shear_range']:
#     #     shear = np.random.uniform(-params['shear_range'], params['shear_range'])
#     # else:
#     #     shear = 0
#     shear = 0
#     shear_matrix = np.array([[1, -np.sin(shear), 0],
#                              [0, np.cos(shear), 0],
#                              [0, 0, 1]])
#
#     if params['zoom_range'][0] == 1 and params['zoom_range'][1] == 1:
#         zx, zy = 1, 1
#     else:
#         zx, zy = np.random.uniform(params['zoom_range'][0], params['zoom_range'][1], 2)
#     zoom_matrix = np.array([[zx, 0, 0],
#                             [0, zy, 0],
#                             [0, 0, 1]])
#
#     transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
#
#     # tweak for off-center transform
#     h, w = int(x.shape[img_row_index]), int(x.shape[img_col_index])
#     transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
#
#     # apply transform matrix - this is hard to do in tensorflow
#     x = apply_transform(x, transform_matrix, img_channel_index,
#                         fill_mode=params['fill_mode'], cval=params['cval'])
#
#     return x


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):

    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [
        ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode=fill_mode,
            cval=cval)
        for x_channel in x]

    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def flip_axis(x, axis):  # TODO: remove?
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
