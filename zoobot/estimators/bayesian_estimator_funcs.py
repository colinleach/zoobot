import logging

import tensorflow as tf
from tensorboard import summary as tensorboard_summary
from tensorflow.python.saved_model import signature_constants

SAMPLES = 10


def four_layer_binary_classifier(features, labels, mode, params):
    """
    Estimator wrapper function for four-layer cnn performing binary classification
    Details (e.g. neurons, activation funcs, etc) controlled by 'params'

    Args:
        features ():
        labels ():
        mode ():
        params ():

    Returns:

    """
    predictions, loss = bayesian_cnn(features, labels, mode, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope('predict'):

            """
            Output a single sample (but calculate many). Deprecated
            """
            # export_outputs = {
            #     'a_name': tf.estimator.export.ClassificationOutput(scores=predictions['probabilities_0'])
            # }

            all_predictions = [predictions['predictions_{}'.format(n)] for n in range(SAMPLES)]
            export_outputs = {
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({
                    'all_predictions': tf.stack(all_predictions, axis=1)  # axis=0 is batch dimension
                    })
            }

            # export_outputs = {}
            # for n in range(SAMPLES):
            #     if n == 0:
            #         name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            #     else:
            #         name = 'sample_{}'.format(n)
            #     export_outputs.update({
            #         name: tf.estimator.export.PredictOutput(
            #             {
            #                 'logits': predictions['probabilities_{}'.format(n)],
            #                 'featured_score': predictions['predictions_{}'.format(n)],
            #             })
            #     })

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.variable_scope('train'):
            optimizer = params['optimizer'](learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    else:  # must be EVAL mode
        with tf.variable_scope('eval'):

            # TODO doesn't work yet
            # tensorboard_summary.pr_curve_streaming_op(
            #     name='spirals',
            #     labels=labels,
            #     predictions=predictions['probabilities'][:, 1],
            # )

            # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = get_eval_metric_ops(labels, predictions)
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def bayesian_cnn(features, labels, mode, params):
    """
    Model function of four-layer CNN
    Can be used in isolation or called within an estimator e.g. four_layer_binary_classifier

    Args:
        features ():
        labels ():
        mode ():
        params ():

    Returns:

    """

    dense1 = input_to_dense(features, params)  # run from input to dense1 output

    # if predict mode, feedforward from dense1 SEVERAL TIMES. Save all predictions under 'all_predictions'.
    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction_tensors = ['predictions', 'probabilities']
        predictions = {}

        for sample in range(SAMPLES):
            # variable names must be unique within a scope - make a new scope for each iteration
            # see https://www.tensorflow.org/programmers_guide/variables#sharing_variables
            with tf.variable_scope("sample_{}".format(sample)):
                # Feedforward from dense1. Always apply dropout.
                _, sample_predictions = dense_to_prediction(dense1, labels, params, dropout_on=True)
                # add to predictions to output, renamed with '_n'
                for tensor in prediction_tensors:
                    predictions[tensor + '_{}'.format(sample)] = sample_predictions[tensor]
                # must return as several dict entries: all dict entries must be the same length (batch_size)

        return predictions, None  # no loss, as labels not known (in general)

    else:  # Calculate Loss for TRAIN and EVAL modes)
        # only feedforward once for one set of predictions
        logits, predictions = dense_to_prediction(dense1, labels, params, dropout_on=mode == tf.estimator.ModeKeys.TRAIN)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        onehot_labels = tf.stop_gradient(onehot_labels)  # don't find the gradient of the labels (e.g. adversarial)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits, name='model/layer4/loss')
        mean_loss = tf.reduce_mean(loss, name='mean_loss')

        # create dummy variables that match names in predict mode
        # TODO this is potentially wasteful as we don't actually need the feedforwards. Unclear if it executes - check.
        for sample in range(SAMPLES):
            with tf.variable_scope("sample_{}".format(sample)):
                _, _ = dense_to_prediction(dense1, labels, params, dropout_on=True)

        return predictions, mean_loss


def input_to_dense(features, params):
    input_layer = features["x"]
    tf.summary.image('model_input', input_layer, 1)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=params['conv1_filters'],
        kernel_size=params['conv1_kernel'],
        padding=params['padding'],
        activation=params['conv1_activation'],
        name='model/layer1/conv1')
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=params['pool1_size'],
        strides=params['pool1_strides'],
        name='model/layer1/pool1')

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=params['conv2_filters'],
        kernel_size=params['conv2_kernel'],
        padding=params['padding'],
        activation=params['conv2_activation'],
        name='model/layer2/conv2')
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=params['pool2_size'],
        strides=params['pool2_strides'],
        name='model/layer2/pool2')

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=params['conv3_filters'],
        kernel_size=params['conv3_kernel'],
        padding=params['padding'],
        activation=params['conv3_activation'],
        name='model/layer3/conv3')
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=params['pool3_size'],
        strides=params['pool3_strides'],
        name='model/layer3/pool3')

    # Flatten tensor into a batch of vectors
    pool3_flat = tf.reshape(pool3, [-1, int(params['image_dim'] / 8) ** 2 * 64], name='model/layer3/flat')

    # Dense Layer
    dense1 = tf.layers.dense(
        inputs=pool3_flat,
        units=params['dense1_units'],
        activation=params['dense1_activation'],
        name='model/layer4/dense1')

    return dense1


def dense_to_prediction(dense1, labels, params, dropout_on):

    # Add dropout operation
    dropout = tf.layers.dropout(
        inputs=dense1,
        rate=params['dense1_dropout'],
        training=dropout_on)
    tf.summary.tensor_summary('dropout_summary', dropout)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=2, name='logits')
    tf.summary.histogram('logits_featured', logits[:, 0])  # first dimension is batch, second is class (0 = featured)
    tf.summary.histogram('logits_smooth', logits[:, 1])

    softmax = tf.nn.softmax(logits, name='softmax')
    softmax_featured_score = softmax[:, 0]
    tf.summary.histogram('softmax_featured_score', softmax_featured_score)

    prediction = {
        "probabilities": softmax,
        "predictions": softmax_featured_score,  # keep only one softmax per subject. Softmax sums to 1 in TF.
    }
    if labels is not None:
        prediction.update({
            'labels': tf.identity(labels, name='labels'),  #  these are None in predict mode
            "classes": tf.argmax(input=logits, axis=1, name='classes'),
        })

    return logits, prediction


def get_eval_metric_ops(labels, predictions):
    # record distribution of predictions for tensorboard
    tf.summary.histogram('Probabilities', predictions['probabilities'])
    tf.summary.histogram('Classes', predictions['classes'])

    return {
        "acc/accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]),
        "acc/mean_per_class_accuracy": tf.metrics.mean_per_class_accuracy(labels=labels,
                                                                          predictions=predictions['classes'],
                                                                          num_classes=2),
        "pr/auc": tf.metrics.auc(labels=labels, predictions=predictions['classes']),
        'pr/precision': tf.metrics.precision(labels=labels, predictions=predictions['classes']),
        'pr/recall': tf.metrics.recall(labels=labels, predictions=predictions['classes']),
        'confusion/true_positives': tf.metrics.true_positives(labels=labels, predictions=predictions['classes']),
        'confusion/true_negatives': tf.metrics.true_negatives(labels=labels, predictions=predictions['classes']),
        'confusion/false_positives': tf.metrics.false_positives(labels=labels, predictions=predictions['classes']),
        'confusion/false_negatives': tf.metrics.false_negatives(labels=labels, predictions=predictions['classes'])
    }


def logging_hooks(params):
    train_tensors = {
        'labels': 'labels',
        # 'logits': 'logits',  may not always exist? TODO
        "probabilities": 'softmax',
        'mean_loss': 'mean_loss'
    }
    train_hook = tf.train.LoggingTensorHook(
        tensors=train_tensors, every_n_iter=params['log_freq'])

    # eval_hook = train_hook
    eval_hook = tf.train.LoggingTensorHook(
        tensors=train_tensors, every_n_iter=params['log_freq'])

    prediction_tensors = {}
    # [prediction_tensors.update({'sample_{}/predictions'.format(n): 'sample_{}/predictions'.format(n)}) for n in range(3)]

    prediction_hook = tf.train.LoggingTensorHook(
        tensors=prediction_tensors, every_n_iter=params['log_freq']
    )

    return [train_hook], [eval_hook], [prediction_hook]  # estimator expects lists of logging hooks
