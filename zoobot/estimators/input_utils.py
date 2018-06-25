from functools import partial

import numpy as np
import tensorflow as tf


from zoobot.tfrecord.tfrecord_io import load_dataset

class InputConfig():

    def __init__(
            self,
            name,
            tfrecord_loc,
            label_col,
            initial_size,
            final_size,
            channels,
            batch_size,
            stratify,
            stratify_probs,
            geometric_augmentation=True,
            shift_range=None,  # not implemented
            max_zoom=1.1,
            fill_mode=None,  # not implemented
            photographic_augmentation=True,
            max_brightness_delta=0.05,
            contrast_range=(0.95, 1.15)
    ):

        self.name = name
        self.tfrecord_loc = tfrecord_loc
        self.label_col = label_col
        self.initial_size = initial_size
        self.final_size = final_size
        self.channels = channels
        self.batch_size = batch_size
        self.stratify = stratify
        self.stratify_probs = stratify_probs

        self.geometric_augmentation = geometric_augmentation  # use geometric augmentations
        self.shift_range = shift_range  # not yet implemented
        self.max_zoom = max_zoom
        self.fill_mode = fill_mode  # not yet implemented, 'pad' or 'zoom'

        self.photographic_augmentation = photographic_augmentation
        self.max_brightness_delta = max_brightness_delta
        self.contrast_range = contrast_range

def get_input(config):
    """
    Load tfrecord as dataset. Stratify and transform_3d images as directed. Batch with queues for Estimator input.
    Args:
        config (InputConfig): Configuration object defining how 'get_input' should function  # TODO consider active class

    Returns:
        (dict) of form {'x': greyscale image batch}, represented as np.float32 Tensor of shape [batch, size, size, 1]}
        (Tensor) categorical labels for each image
    """
    with tf.name_scope('input_{}'.format(config.name)):
        feature_spec = matrix_label_feature_spec(config.initial_size, config.channels)

        # 'think of this as a lazy list of tuples of tensors'
        dataset = load_dataset(config.tfrecord_loc, feature_spec, num_parallel_calls=1)
        dataset = dataset.shuffle(config.batch_size * 10)
        # dataset = dataset.repeat(5)
        dataset = dataset.batch(config.batch_size)
        # dataset.map(func, num_parallel_calls=n)?
        dataset = dataset.prefetch(1)  # ensure that 1 batch is always ready to go
        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        batch_images, batch_labels = batch['matrix'], batch['label']

        if config.stratify:
            #  warning - stratify only works if initial probabilities are specified
            batch_images_stratified, batch_labels_stratified = stratify_images(
                batch_images,
                batch_labels,
                config.batch_size,
                config.stratify_probs
            )

            batch_images = batch_images_stratified
            batch_labels = batch_labels_stratified

        else:
            assert config.stratify_probs is None  # check in case user has accidentally forgotten to activate stratify
            # TODO make a single stratify parameter that expects list of floats - required to run properly anyway

        preprocessed_batch_images = preprocess_batch(batch_images, config)

        return preprocessed_batch_images, batch_labels


def preprocess_batch(batch_images, input_config):
    with tf.name_scope('preprocess'):
        batch_images = tf.reshape(
            batch_images,
            [-1, input_config.initial_size, input_config.initial_size,
             input_config.channels])
        tf.summary.image('preprocess/original', batch_images)

        batch_images = tf.reduce_mean(batch_images, axis=3, keep_dims=True)
        tf.summary.image('preprocess/greyscale', batch_images)

        batch_images = augment_images(batch_images, input_config)
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



def matrix_label_feature_spec(size, channels):
    return {
        "matrix": tf.FixedLenFeature((size * size * channels), tf.float32),
        "label": tf.FixedLenFeature((), tf.int64)}


def matrix_feature_spec(size, channels):  # used for predict mode
    return {
        "matrix": tf.FixedLenFeature((size * size * channels), tf.float32)}



def augment_images(images, input_config):
    """

    Args:
        images (tf.Variable):
        input_config (InputConfig):

    Returns:

    """
    # image_list = tf.unstack(images, axis=0)
    if input_config.geometric_augmentation:
        images = tf.map_fn(
            partial(
                geometric_augmentation,
                max_zoom=input_config.max_zoom,
                final_size=input_config.final_size
            ),
            images,
            back_prop=False
        )
    if input_config.photographic_augmentation:
        images = tf.map_fn(
            partial(
                photographic_augmentation,
                max_brightness_delta=input_config.max_brightness_delta,
                contrast_range=input_config.contrast_range),
            images)
    return images


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


def geometric_augmentation(image, max_zoom, final_size):
    """
    Runs best if image is originally significantly larger than final target size
    for example: load at 256px, rotate/flip, crop to 246px, then finally resize to 64px
    more computation but more pixel info preserved

    Args:
        image ():
        max_zoom ():
        final_size ():

    Returns:
        (Tensor): image rotated, flipped, cropped and (perhaps) normalized, shape (target_size, target_size, channels)
    """
    print('Applying augmentation')
    assert image.shape[0] == image.shape[1]  # must be square
    assert len(image.shape) == 3 # must have no batch dimension
    assert max_zoom > 1. and max_zoom < 10.  # catch user accidentally putting in pixel values here
    # TODO add slight redshifting?
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    rotation_angle_radians = np.random.rand() * np.pi
    print('Angle: ', rotation_angle_radians)
    image = tf.contrib.image.rotate(
        image,
        rotation_angle_radians,  # in radians
        interpolation='NEAREST',  # Estimate new pixel values with interpolation. Empty space is filled with zeros.
    )  # TODO file tensorflow issue: fails on interpolation=bilinear
    # TODO add stretch and/or shear?
    # TODO add Gaussian or Poisson noise?
    # crop down and leave it small. Size is absolute.

    crop_size = int(int(image.shape[0]) / np.random.uniform(1.0, max_zoom))  # if max_zoom = 1.3, zoom randomly 1x to 1.3x
    image = tf.random_crop(image, [crop_size, crop_size, image.shape[2]])  # do not change 'channel' dimension
    # resize to final desired size (may match crop size)
    # pad...
    # image = tf.image.resize_image_with_crop_or_pad(
    #     image,
    #     target_size,
    #     target_size
    # )
    # ...or zoom
    image = tf.image.resize_images(  # bilinear by default, could do bicubic
        image,
        tf.constant([final_size, final_size], dtype=tf.int32),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR  # again, only nearest neighbour works - otherwise gives noise
    )
    return image


def photographic_augmentation(image, max_brightness_delta, contrast_range):
    image = tf.image.random_brightness(image, max_delta=max_brightness_delta)
    image = tf.image.random_contrast(image, lower=contrast_range[0], upper=contrast_range[1])
    # Subtract off the mean and divide by the variance of the pixels
    # image = tf.image.per_image_standardization(image)   # TODO consider astro-suitable normalization
    return image


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
#
#
# def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
#
#     x = np.rollaxis(x, channel_index, 0)
#     final_affine_matrix = transform_matrix[:2, :2]
#     final_offset = transform_matrix[:2, 2]
#     channel_images = [
#         ndimage.interpolation.affine_transform(
#             x_channel,
#             final_affine_matrix,
#             final_offset,
#             order=0,
#             mode=fill_mode,
#             cval=cval)
#         for x_channel in x]
#
#     x = np.stack(channel_images, axis=0)
#     x = np.rollaxis(x, 0, channel_index + 1)
#     return x
#
#
# def transform_matrix_offset_center(matrix, x, y):
#     o_x = float(x) / 2 + 0.5
#     o_y = float(y) / 2 + 0.5
#     offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
#     reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
#     transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
#     return transform_matrix
#
#
# def flip_axis(x, axis):  # TODO: remove?
#     x = np.asarray(x).swapaxes(axis, 0)
#     x = x[::-1, ...]
#     x = x.swapaxes(0, axis)
#     return x
