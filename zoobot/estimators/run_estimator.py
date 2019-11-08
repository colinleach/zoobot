import logging
import os
import shutil

import copy
from functools import partial

import numpy as np
import tensorflow as tf
from zoobot.estimators import input_utils, bayesian_estimator_funcs

# don't decorate, this is session creation point
def run_estimator(config):
    """
    Train and evaluate an estimator.
    `config` may well be provided by default_estimator_params.py`

    TODO save every n epochs 
    TODO enable early stopping
    TODO enable use with tf.serving
    TODO enable logging hooks?

    Args:
        config (RunEstimatorConfig): parameters controlling both estimator and train/test procedure

    Returns:
        None
    """
    if not config.warm_start:  # don't try to load any existing models
        if os.path.exists(config.log_dir):
            shutil.rmtree(config.log_dir)

    train_dataset = input_utils.get_input(config=config.train_config)
    test_dataset = input_utils.get_input(config=config.eval_config)

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.log_dir, 'tensorboard'),
            histogram_freq=1,
            write_images=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.log_dir, 'models'),
            save_weights_only=True)
    ]

    # if this doesn't get the steps right, can use any custom callback to keras.backend.set_value self.epoch=epoch, and then read that on each summary call
    # if this doesn't get train/test right, could similarly use the callbacks to set self.mode

    fit_summary_writer = tf.summary.create_file_writer(os.path.join(config.log_dir, 'manual_summaries'))
    with fit_summary_writer.as_default():
        config.model.fit(
            train_dataset,
            validation_data=test_dataset,
            validation_steps=10,
            epochs=config.epochs,
            callbacks=callbacks,
            metrics=[
                bayesian_estimator_funcs.custom_smooth_mse,
                bayesian_estimator_funcs.custom_spiral_mse
            ]
        )

    logging.info('All epochs completed - finishing gracefully')
    config.model.save_weights(os.path.join(config.log_dir, 'models/final'))
    return config.model
