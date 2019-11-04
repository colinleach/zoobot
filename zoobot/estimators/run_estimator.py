import logging
import os
import shutil

import copy
from functools import partial

import numpy as np
import tensorflow as tf
from zoobot.estimators import input_utils, bayesian_estimator_funcs


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

    config.model.fit(
        train_dataset,
        validation_data=test_dataset,
        validation_steps=10,
        epochs=config.epochs
    )

    logging.info('All epochs completed - finishing gracefully')
    config.model.save_weights(config.log_dir)
    return config.model
