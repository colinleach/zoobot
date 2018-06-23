import logging
import os
import shutil
import time
from functools import partial

import numpy as np
import tensorflow as tf
from zoobot.estimators import input_utils, bayesian_estimator_funcs


class RunEstimatorConfig():

    def __init__(
            self,
            image_dim,
            channels,
            label_col,
            epochs=50,
            train_batches=30,
            eval_batches=3,
            batch_size=128,
            min_epochs=0,
            early_stopping_window=10,
            max_sadness=4.,
            log_dir='runs/default_run_{}'.format(time.time()),
            save_freq=10
    ):
        self.image_dim = image_dim
        self.channels = channels
        self.label_col = label_col
        self.epochs = epochs
        self.train_batches = train_batches
        self.eval_batches = eval_batches
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.max_sadness = max_sadness
        self.early_stopping_window = early_stopping_window
        self.min_epochs = min_epochs
        self.train_config = None
        self.eval_config = None
        self.model = None

    def is_ready_to_train(self):
        # TODO can make this check much more comprehensive
        return (self.train_config is not None) and (self.eval_config is not None)


def train_input(run_config):
    return input_utils.get_input(config=run_config.train_config)


def eval_input(run_config):
    return input_utils.get_input(config=run_config.eval_config)


def run_estimator(config):
    """
    Train and evaluate an estimator

    Args:
        model_fn (function): estimator model function
        config (RunEstimatorConfig): parameters controlling both estimator and train/test procedure

    Returns:
        None
    """
    assert config.is_ready_to_train()

    # start fresh, don't try to load any existing models
    if os.path.exists(config.log_dir):
        shutil.rmtree(config.log_dir)

    # Create the Estimator
    model_fn_partial = partial(bayesian_estimator_funcs.estimator_wrapper)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn_partial,
        model_dir=config.log_dir,
        params=config.model
    )

    # can't move out of run_estimator, uses closure to avoid arguments
    def serving_input_receiver_fn():
        """An input receiver that expects a serialized tf.Example."""
        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                               name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        feature_spec = input_utils.matrix_feature_spec(size=config.image_dim, channels=config.channels)
        features = tf.parse_example(serialized_tf_example, feature_spec)
        # update each image with the preprocessing from input_utils
        # outputs {x: new images}

        # TODO DANGER this can deviate from input utils - cause of bug?
        new_features = input_utils.preprocess_batch(
            features['matrix'],
            input_config=config.eval_config
        )
        return tf.estimator.export.ServingInputReceiver(new_features, receiver_tensors)

    train_input_partial = partial(train_input, params=config)
    eval_input_partial = partial(eval_input, params=config)

    train_logging, eval_logging, predict_logging = config.model.logging_hooks

    eval_loss_history = []
    epoch_n = 0

    while epoch_n <= config.epochs:
        # Train the estimator
        estimator.train(
            input_fn=train_input_partial,
            steps=config.train_batches,
            # hooks=train_logging + [tf.train.ProfilerHook(save_secs=10)]
            hooks=train_logging
        )

        # Evaluate the estimator and logging.info results
        eval_metrics = estimator.evaluate(
            input_fn=eval_input_partial,
            steps=config.eval_batches,
            hooks=eval_logging
        )
        eval_loss_history.append(eval_metrics['loss'])

        # predictions = estimator.predict(
        #     eval_input_partial,
        #     hooks=predict_logging
        # )
        # prediction_rows = list(predictions)
        # logging.debug('Predictions ({}): '.format(len(prediction_rows)))
        # for row in prediction_rows[:10]:
        #     logging.info(row)

        if epoch_n % config.save_freq == 0:
            save_model(estimator, config, epoch_n, serving_input_receiver_fn)

        if epoch_n > config.min_epochs:
            sadness = early_stopper(eval_loss_history, config.early_stopping_window)
            logging.info('Current sadness: {}'.format(sadness))
            if sadness > config.max_sadness:
                logging.info('Ending training at epoch {} with {} sadness'.format(
                    epoch_n,
                    sadness))
                break  # stop training early

        logging.info('End epoch {}'.format(epoch_n))
        epoch_n += 1

    logging.info('All epochs completed - finishing gracefully')

    return eval_loss_history


def loss_instability(loss_history, window):
    return (np.mean(loss_history[-window:]) / np.min(loss_history[-window:])) - 1


def generalised_loss(loss_history):
    return (loss_history[-1] / np.min(loss_history)) - 1


def early_stopper(loss_history, window):
    return generalised_loss(loss_history) / loss_instability(loss_history, window)


def save_model(estimator, config, epoch_n, serving_input_receiver_fn):
    """

    Args:
        estimator:
        config (R:
        epoch_n:
        serving_input_receiver_fn:

    Returns:

    """
    logging.info('Saving model at epoch {}'.format(epoch_n))
    estimator.export_savedmodel(
        export_dir_base=config.log_dir,
        serving_input_receiver_fn=serving_input_receiver_fn)
