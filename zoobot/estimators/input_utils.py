import copy

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
            shuffle,
            stratify,
            stratify_probs,
            geometric_augmentation=True,
            shift_range=None,  # not implemented
            max_zoom=1.1,
            fill_mode=None,  # not implemented
            photographic_augmentation=True,
            max_brightness_delta=0.05,
            contrast_range=(0.95, 1.05)
    ):

        self.name = name
        self.tfrecord_loc = tfrecord_loc
        self.label_col = label_col
        self.initial_size = initial_size
        self.final_size = final_size
        self.channels = channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stratify = stratify
        self.stratify_probs = stratify_probs

        self.geometric_augmentation = geometric_augmentation  # use geometric augmentations
        self.shift_range = shift_range  # not yet implemented
        self.max_zoom = max_zoom
        self.fill_mode = fill_mode  # not yet implemented, 'pad' or 'zoom'

        self.photographic_augmentation = photographic_augmentation
        self.max_brightness_delta = max_brightness_delta
        self.contrast_range = contrast_range

    # TODO move to shared utilities
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])

    def copy(self):
        return copy.deepcopy(self)


def get_input(config):
    """
    Load tfrecord as dataset. Stratify and transform_3d images as directed. Batch with queues for Estimator input.
    Args:
        config (InputConfig): Configuration object defining how 'get_input' should function  # TODO consider active class

    Returns:
        (dict) of form {'x': greyscale image batch}, as Tensor of shape [batch, size, size, 1]}
        (Tensor) categorical labels for each image
    """
    with tf.name_scope('input_{}'.format(config.name)):
        batch_images, batch_labels = load_batches(config)
        preprocessed_batch_images = preprocess_batch(batch_images, config)
        # tf.shape is important to record the dynamic shape, rather than static shape
        tf.summary.scalar('batch_size', tf.shape(preprocessed_batch_images['x'])[0])
        tf.summary.scalar('mean_label', tf.reduce_mean(batch_labels))
        return preprocessed_batch_images, batch_labels


def load_batches(config):
    """
    Get batches of images and labels from tfrecord according to instructions in config
    # TODO make a single stratify parameter that expects list of floats - required to run properly anyway
    # use e.g. dataset.map(func, num_parallel_calls=n) rather than map_fn - but stratify??

    Args:
        config (InputConfig): instructions to load and preprocess the image and label data

    Returns:
        (tf.Tensor, tf.Tensor)
    """
    with tf.name_scope('load_batches_{}'.format(config.name)):
        feature_spec = matrix_label_feature_spec(config.initial_size, config.channels)

        dataset = load_dataset(config.tfrecord_loc, feature_spec)

        if config.shuffle:
            dataset = dataset.shuffle(config.batch_size * 5)
        # dataset = dataset.repeat(5)  Careful, don't repeat for eval - make param
        dataset = dataset.batch(config.batch_size)
        dataset = dataset.prefetch(1)  # ensure that 1 batch is always ready to go
        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        batch_data, batch_labels = batch['matrix'], batch['label']

        if config.stratify:
            #  warning - stratify only works if initial probabilities are specified
            batch_images_stratified, batch_labels_stratified = stratify_images(
                batch_data,
                batch_labels,
                config.batch_size,
                config.stratify_probs
            )
            batch_data = batch_images_stratified
            batch_labels = batch_labels_stratified
        else:
            assert config.stratify_probs is None  # check in case user has accidentally forgotten to activate stratify

        batch_images = tf.reshape(
            batch_data,
            [-1, config.initial_size, config.initial_size,
             config.channels])
        tf.summary.image('a_original', batch_images)

    assert len(batch_images.shape) == 4
    return batch_images, batch_labels


def preprocess_batch(batch_images, config):
    with tf.name_scope('preprocess'):

        assert len(batch_images.shape) == 4

        greyscale_images = tf.reduce_mean(batch_images, axis=3, keepdims=True)   # new channel dimension of 1
        assert greyscale_images.shape[1] == config.initial_size
        tf.summary.image('b_greyscale', greyscale_images)

        augmented_images = augment_images(greyscale_images, config)
        tf.summary.image('c_augmented', augmented_images)

        feature_cols = {'x': augmented_images}
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
    if input_config.geometric_augmentation:
        images = geometric_augmentation(
            images,
            max_zoom=input_config.max_zoom,
            final_size=input_config.final_size)

    if input_config.photographic_augmentation:
        images = photographic_augmentation(
            images,
            max_brightness_delta=input_config.max_brightness_delta,
            contrast_range=input_config.contrast_range)

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


def geometric_augmentation(images, max_zoom, final_size):
    """
    Runs best if image is originally significantly larger than final target size
    for example: load at 256px, rotate/flip, crop to 246px, then finally resize to 64px
    This leads to more computation, but more pixel info is preserved

    # TODO add stretch and/or shear?

    Args:
        images ():
        max_zoom ():
        final_size ():

    Returns:
        (Tensor): image rotated, flipped, cropped and (perhaps) normalized, shape (target_size, target_size, channels)
    """

    images = ensure_images_have_batch_dimension(images)

    assert images.shape[1] == images.shape[2]  # must be square
    assert max_zoom > 1. and max_zoom < 10.  # catch user accidentally putting in pixel values here

    # flip functions don't support batch dimension - wrap with map_fn
    images = tf.map_fn(
        tf.image.random_flip_left_right,
        images)
    images = tf.map_fn(
        tf.image.random_flip_up_down,
        images)
    images = tf.map_fn(
        random_rotation,
        images)

    # if max_zoom = 1.3, zoom randomly 1x to 1.3x
    images = tf.map_fn(lambda x: random_crop_random_size(x, max_zoom=max_zoom), images)

    # do not change batch  or 'channel' dimension
    # resize to final desired size (may match crop size)
    # pad...
    # image = tf.image.resize_image_with_crop_or_pad(
    #     image,
    #     target_size,
    #     target_size
    # )
    # ...or zoom
    images = tf.image.resize_images(
        images,
        tf.constant([final_size, final_size], dtype=tf.int32),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR  # only nearest neighbour works - otherwise gives noise
    )
    return images


def random_rotation(im):
    return tf.contrib.image.rotate(
        im,
        3.14 * tf.random_uniform(shape=[1]),
        interpolation='BILINEAR'
    )


def random_crop_random_size(im, max_zoom):
    new_width = int(int(im.shape[1]) / np.random.uniform(1.0, max_zoom))  # first int cast allows division of Dimension
    cropped_shape = tf.constant([new_width, new_width, int(im.shape[2])], dtype=tf.int32)
    return tf.random_crop(im, cropped_shape)


def photographic_augmentation(images, max_brightness_delta, contrast_range):
    """
    TODO do before or after geometric?
        TODO add slight redshifting?
    TODO

    Args:
        images ():
        max_brightness_delta ():
        contrast_range ():

    Returns:

    """
    images = ensure_images_have_batch_dimension(images)

    images = tf.map_fn(
        lambda im: tf.image.random_brightness(im, max_delta=max_brightness_delta),
        images)
    images = tf.map_fn(
        lambda im: tf.image.random_contrast(im, lower=contrast_range[0], upper=contrast_range[1]),
        images)

    return images


def ensure_images_have_batch_dimension(images):
    if len(images.shape) < 3:
        raise ValueError
    if len(images.shape) == 3:
        images = tf.expand_dims(images, axis=0)  # add a batch dimension
    return images
