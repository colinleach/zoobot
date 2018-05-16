import logging
import os
import shutil
from functools import partial

import tensorflow as tf
from zoobot.estimators import input_utils


# def four_layer_regression_classifier(features, labels, mode, params):
#     """
#     Classify images of galaxies into spiral/not spiral
#
#     Args:
#         features ():
#         labels ():
#         mode ():
#         params ():
#
#     Returns:
#
#     """
#
#     predictions, loss = four_layer_cnn(features, labels, mode, params)  # loss is None if in predict mode
#
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
#
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         optimizer = params['optimizer'](learning_rate=params['learning_rate'])
#         train_op = optimizer.minimize(
#             loss=loss,
#             global_step=tf.train.get_global_step())
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
#
#     # tensorboard_summary.pr_curve_streaming_op(
#     #     name='spirals',
#     #     labels=labels,
#     #     predictions=predictions['probabilities'][:, 1],
#     # )
#     # Add evaluation metrics (for EVAL mode)
#     eval_metric_ops = get_eval_metric_ops(labels, predictions)
#     return tf.estimator.EstimatorSpec(
#         mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train_input(params):
    mode = 'train'
    return input_utils.input(
        tfrecord_loc=params['train_loc'], size=params['image_dim'], name=mode, batch_size=params['batch_size'], stratify=params['train_stratify'])


def eval_input(params):
    mode = 'test'
    return input_utils.input(
        tfrecord_loc=params['test_loc'], size=params['image_dim'], name=mode, batch_size=params['batch_size'], stratify=params['eval_stratify'])


# TODO wrap estimator for serving (maybe not here)
# def serving_input_receiver_fn():
#     """Build the serving inputs."""
#     # The outer dimension (None) allows us to batch up inputs for
#     # efficiency. However, it also means that if we want a prediction
#     # for a single instance, we'll need to wrap it in an outer list.
#     inputs = {"x": tf.placeholder(shape=[None, 4], dtype=tf.float32)}
#     return tf.estimator.export.ServingInputReceiver(inputs, inputs)

#
# def serving_input_receiver_fn():
#   """An input receiver that expects a serialized tf.Example."""
#   serialized_tf_example = tf.placeholder(dtype=tf.string,
#                                          shape=[default_batch_size],
#                                          name='input_example_tensor')
#   receiver_tensors = {'examples': serialized_tf_example}
#   features = tf.parse_example(serialized_tf_example, matrix_label_feature_spec)
#   return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def run_estimator(model_fn, params):
    """
    Train and evaluate an estimator

    Args:
        model_fn (function): estimator model function
        params (dict): parameters controlling both estimator and train/test procedure

    Returns:
        None
    """

    if os.path.exists(params['log_dir']):
        shutil.rmtree(params['log_dir'])

    # Create the Estimator
    model_fn_partial = partial(model_fn, params=params)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn_partial, model_dir=params['log_dir'])

    # Set up logging for predictions
    # Apparently, looks at tensors output under 'predictions', not anywhere within graph? unclear

    train_input_partial = partial(train_input, params=params)
    eval_input_partial = partial(eval_input, params=params)

    train_logging, eval_logging, predict_logging = params['logging_hooks']

    epoch_n = 0
    while epoch_n < params['epochs']:
        logging.info('training begins')
        # Train the estimator
        estimator.train(
            input_fn=train_input_partial,
            steps=params['train_batches'],
            max_steps=params['max_train_batches'],
            hooks=train_logging
        )

        # Evaluate the estimator and logging.info results
        logging.info('eval begins')
        _ = estimator.evaluate(
            input_fn=eval_input_partial,
            steps=params['eval_batches'],
            hooks=eval_logging
        )

        # TODO
        # logging.info('Saving model at epoch {}'.format(epoch_n))
        # estimator.export_savedmodel(
        #     export_dir_base="/path/to/model",
        #     serving_input_receiver_fn=serving_input_receiver_fn)

        predictions = estimator.predict(
            eval_input_partial,
            hooks=predict_logging
        )
        logging.info([n.name for n in tf.get_default_graph().as_graph_def().node])
        prediction_rows = list(predictions)
        logging.info('Predictions: ')
        logging.info(len(prediction_rows))
        for row in prediction_rows[:10]:
            logging.info(row)

        epoch_n += 1
