import os
import shutil

import tensorflow as tf
import numpy as np

from read_tfrecord import read_and_decode_single_example

SIZE = 64

#
# def my_input_function():
#     """
#     Estimator input returns {col: values} and labels
#     Symbolic, many examples
#     Returns:
#
#     """
#     # filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_64.tfrecord'
#     filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_28.tfrecord'
#     label, image, spiral_fraction = read_and_decode_single_example(filename)
#     labels_batch, images_batch, spiral_fraction_batch = tf.train.shuffle_batch(
#         [label, image, spiral_fraction], batch_size=100,
#         capacity=1000,
#         min_after_dequeue=100)
#     feature_cols = {'x': images_batch}
#     return feature_cols, labels_batch


def my_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = features["x"]

    tf.summary.image('augmented', input_layer, 1)

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64], name='pool2_flat')
    pool2_flat = tf.reshape(pool2, [-1, int(SIZE/4) ** 2 * 64], name='pool2_flat')
    # tf.Print(pool2_flat, [pool2_flat.shape])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name='dense')
    # tf.Print(dense, [dense.shape])

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    tf.summary.tensor_summary('dropout_summary', dropout)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    # logits = tf.layers.dense(inputs=dropout, units=10)
    logits = tf.layers.dense(inputs=dropout, units=2)
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('logits probabilities', tf.nn.softmax(logits))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# TODO combine train and eval functions to avoid inconsistency

# 191756 examples
def train_input():
    filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_train.tfrecord'.format(SIZE)

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.shuffle(190000)
    dataset = dataset.batch(256)
    iterator = dataset.make_one_shot_iterator()
    batch_images, batch_labels = iterator.get_next()
    batch_images = tf.reshape(batch_images, [-1, SIZE, SIZE, 3])
    tf.summary.image('original', batch_images)
    grey_images = tf.reduce_mean(batch_images, axis=3, keep_dims=True)
    tf.summary.image('greyscale', batch_images)
    augmented_images = augment_images(grey_images)

    # TODO new and untested
    # (stratified_images, stratified_labels) = tf.contrib.training.stratified_sample(
    #     augmented_images,
    #     batch_labels,
    #     target_probs=[0.5, 0.5],
    #     batch_size=128)

    feature_cols = {'x': augmented_images}
    return feature_cols, batch_labels


def eval_input():
    filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_test.tfrecord'.format(SIZE)

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    dataset = dataset.shuffle(47500)
    dataset = dataset.batch(2000)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    batch_images, batch_labels = iterator.get_next()
    batch_images = tf.reshape(batch_images, [-1, SIZE, SIZE, 3])
    tf.summary.image('original_eval', batch_images)
    grey_images = tf.reduce_mean(batch_images, axis=3, keep_dims=True)
    tf.summary.image('greyscale_eval', batch_images)
    augmented_images = augment_images(grey_images)

    # TODO new and untested
    # (stratified_images, stratified_labels) = tf.contrib.training.stratified_sample(
    #     augmented_images,
    #     batch_labels,
    #     target_probs=[0.5, 0.5],
    #     batch_size=1000)

    feature_cols = {'x': augmented_images}
    return feature_cols, batch_labels


def augment_images(images):
    images = tf.image.random_brightness(images, max_delta=0.1)
    images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
    return images


def _parse_function(example_proto):
  features = {"matrix": tf.FixedLenFeature((SIZE * SIZE * 3), tf.float32),
              "label": tf.FixedLenFeature((), tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["matrix"], parsed_features["label"]


def main():
    log_dir = 'mine'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=my_model_fn, model_dir='mine')

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    epoch_n = 0
    while epoch_n < 40:
        # Train the model
        mnist_classifier.train(
            input_fn=train_input,
            steps=50,
            hooks=[logging_hook]
            )
        # Evaluate the model and print results
        eval_results = mnist_classifier.evaluate(input_fn=eval_input)
        print(eval_results)

        epoch_n += 1


if __name__ == '__main__':
    main()
