import os
import shutil

import tensorflow as tf
import numpy as np

from tensorboard import summary as tensorboard_summary
from input_utils import input


SIZE = 64


def my_model_fn(features, labels, mode):
    """Model function for CNN."""
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
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # tf.Print(conv1, [tf.shape(conv1)])
    # print(conv1)

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

    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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


def main():
    log_dir = 'runs/strat_with_augs_64'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=my_model_fn, model_dir=log_dir)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=27)

    def train_input():
        mode = 'train'
        # filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_{}.tfrecord'.format(SIZE, 'test')
        filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_{}.tfrecord'.format(SIZE, mode)
        return input(
           filename=filename, size=SIZE, mode=mode, batch=100, augment=True, stratify=True)

    def eval_input():
        mode = 'test'
        # filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_{}.tfrecord'.format(SIZE, 'train')
        filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_{}.tfrecord'.format(SIZE, mode)
        return input(
            filename=filename, size=SIZE, mode=mode, batch=100, augment=True, stratify=True)

    epoch_n = 0
    while epoch_n < 4000:
        print('training begins')
        # Train the model
        mnist_classifier.train(
            input_fn=train_input,
            steps=3,
            hooks=[logging_hook]
        )

        # result = mnist_classifier.predict(input_fn=eval_input)
        # print(list(result))

        # Evaluate the model and print results
        print('eval begins')
        eval_results = mnist_classifier.evaluate(input_fn=eval_input)
        print(eval_results)

        epoch_n += 1


if __name__ == '__main__':
    main()
