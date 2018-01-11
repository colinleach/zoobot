import os
import shutil

import tensorflow as tf
import numpy as np
from functools import partial

from tensorboard import summary as tensorboard_summary
from input_utils import input


def spiral_classifier(features, labels, mode, params):
    """
    Classify images of galaxies into spiral/not spiral
    Based on MNIST example from tensorflow docs

    Args:
        features ():
        labels ():
        mode ():

    Returns:

    """
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = features["x"]
    # input_layer = tf.Print(input_layer, [input_layer, tf.shape(input_layer)])

    tf.summary.image('augmented', input_layer, 1)

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=params['conv1_filters'],
        kernel_size=params['conv1_kernel'],
        padding=params['conv1_padding'],
        activation=params['conv1_activation'],
        name='model/layer1/conv1')

    # tf.Print(conv1, [tf.shape(conv1)])
    # print(conv1)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=params['pool1_size'],
        strides=params['pool1_strides'],
        name='model/layer1/pool1')

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=params['conv2_filters'],
        kernel_size=params['conv2_kernel'],
        padding=params['conv2_padding'],
        activation=params['conv2_activation'],
        name='model/layer2/conv2')

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=params['pool2_size'],
        strides=params['pool2_strides'],
        name='model/layer2/pool2')

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64], name='pool2_flat')
    pool2_flat = tf.reshape(pool2, [-1, int(params['image_dim']/4) ** 2 * 64], name='model/layer2/flat')
    # tf.Print(pool2_flat, [pool2_flat.shape])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(
        inputs=pool2_flat,
        units=params['dense1_units'],
        activation=params['dense1_activation'],
        name='model/layer3/dense1')
    # tf.Print(dense, [dense.shape])

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense1,
        rate=params['dense1_dropout'],
        training=mode == tf.estimator.ModeKeys.TRAIN)
    tf.summary.tensor_summary('dropout_summary', dropout)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    # logits = tf.layers.dense(inputs=dropout, units=10)
    logits = tf.layers.dense(inputs=dropout, units=2, name='model/layer4/logits')
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
    # required for EstimatorSpec
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params['optimizer'](learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    tf.summary.histogram('Probabilities', predictions['probabilities'])
    tf.summary.histogram('Classes', predictions['classes'])

    tensorboard_summary.pr_curve_streaming_op(
        name='spirals',
        labels=labels,
        predictions=predictions['probabilities'][:, 1],
    )

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "acc/accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]),
        "pr/auc": tf.metrics.auc(labels=labels, predictions=predictions['classes']),
        "acc/mean_per_class_accuracy": tf.metrics.mean_per_class_accuracy(labels=labels, predictions=predictions['classes'], num_classes=2),
        'pr/precision': tf.metrics.precision(labels=labels, predictions=predictions['classes']),
        'pr/recall': tf.metrics.recall(labels=labels, predictions=predictions['classes']),
        'confusion/true_positives': tf.metrics.true_positives(labels=labels, predictions=predictions['classes']),
        'confusion/true_negatives': tf.metrics.true_negatives(labels=labels, predictions=predictions['classes']),
        'confusion/false_positives': tf.metrics.false_positives(labels=labels, predictions=predictions['classes']),
        'confusion/false_negatives': tf.metrics.false_negatives(labels=labels, predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def run_experiment(model_fn, params):

    if os.path.exists(params['log_dir']):
        shutil.rmtree(params['log_dir'])

    # Create the Estimator
    model_fn_partial = partial(model_fn, params=params)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn_partial, model_dir=params['log_dir'])

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=params['log_freq'])

    def train_input():
        mode = 'train'
        # filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_{}.tfrecord'.format(params['image_dim'], 'test')
        filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_{}.tfrecord'.format(params['image_dim'], mode)
        return input(
           filename=filename, size=params['image_dim'], mode=mode, batch=100, augment=True, stratify=True)

    def eval_input():
        mode = 'test'
        # filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_{}.tfrecord'.format(SIZE, 'train')
        filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_{}.tfrecord'.format(params['image_dim'], mode)
        return input(
            filename=filename, size=params['image_dim'], mode=mode, batch=100, augment=True, stratify=True)

    epoch_n = 0
    while epoch_n < params['epochs']:
        print('training begins')
        # Train the estimator
        estimator.train(
            input_fn=train_input,
            steps=params['save_steps'],
            hooks=[logging_hook]
        )

        # result = estimator.predict(input_fn=eval_input)
        # print(list(result))

        # Evaluate the estimator and print results
        print('eval begins')
        eval_results = estimator.evaluate(input_fn=eval_input)
        print(eval_results)

        epoch_n += 1


def default_model_architecture():
    return dict(
        conv1_filters=32,
        conv1_kernel=[5, 5],
        conv1_padding='same',
        conv1_activation=tf.nn.relu,

        pool1_size=[2, 2],
        pool1_strides=2,

        conv2_filters=64,
        conv2_kernel=[5, 5],
        conv2_padding='same',
        conv2_activation=tf.nn.relu,

        pool2_size=[2, 2],
        pool2_strides=2,

        dense1_units=1064,
        dense1_dropout=0.4,
        dense1_activation=tf.nn.relu,

        learning_rate=0.001,
        optimizer=tf.train.GradientDescentOptimizer

    )


def chollet_model_architecture():
    return dict(
        conv1_filters=32,
        conv1_kernel=[3, 3],
        conv1_padding='same',
        conv1_activation=tf.nn.relu,

        pool1_size=[2, 2],
        pool1_strides=1,

        conv2_filters=64,
        conv2_kernel=[3, 3],
        conv2_padding='same',
        conv2_activation=tf.nn.relu,

        pool2_size=[2, 2],
        pool2_strides=1,

        dense1_units=1064,
        dense1_dropout=0.5,
        dense1_activation=tf.nn.relu,

        learning_rate=0.001,
        optimizer=tf.train.GradientDescentOptimizer

    )


def main():

    params = dict(
        epochs=1000,
        batch_size=128,  # TODO
        image_dim=64,
        save_steps=10,
        log_freq=25,
        log_dir='runs/chollet_run'
    )

    # params.update(default_model_architecture())
    params.update(chollet_model_architecture())
    # print(params)

    run_experiment(spiral_classifier, params)


if __name__ == '__main__':
    main()
