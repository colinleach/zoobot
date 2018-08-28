import logging
import os

import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from zoobot.estimators import bayesian_estimator_funcs, run_estimator, input_utils, warm_start


def setup(run_name, train_tfrecord_loc, eval_tfrecord_loc, initial_size, final_size, label_split_value, log_dir):

    channels = 3

    run_config = run_estimator.RunEstimatorConfig(
        initial_size=initial_size,
        final_size=final_size,
        channels=channels,
        label_col='label',
        epochs=5,
        train_steps=5,
        eval_steps=3,
        batch_size=128,
        min_epochs=1000,  # don't stop early automatically, wait for me
        early_stopping_window=10,
        max_sadness=4.,
        log_dir=log_dir,
        save_freq=10,
        fresh_start=False  # Will restore previous run from disk, if saved
    )

    train_config = input_utils.InputConfig(
        name='train',
        tfrecord_loc=train_tfrecord_loc,
        label_col=run_config.label_col,
        stratify=True,
        shuffle=True,
        stratify_probs=None,
        geometric_augmentation=True,
        photographic_augmentation=True,
        max_zoom=1.2,
        fill_mode='wrap',
        batch_size=run_config.batch_size,
        initial_size=run_config.initial_size,
        final_size=run_config.final_size,
        channels=run_config.channels,
    )
    train_config.stratify_probs = get_stratify_probs_from_csv(train_config)

    eval_config = input_utils.InputConfig(
        name='eval',
        tfrecord_loc=eval_tfrecord_loc,
        label_col=run_config.label_col,
        stratify=True,
        shuffle=True,
        stratify_probs=None,
        geometric_augmentation=True,
        photographic_augmentation=True,
        max_zoom=1.2,
        fill_mode='wrap',
        batch_size=run_config.batch_size,
        initial_size=run_config.initial_size,
        final_size=run_config.final_size,
        channels=run_config.channels,
    )
    eval_config.stratify_probs = get_stratify_probs_from_csv(train_config)  # eval not allowed!

    model = bayesian_estimator_funcs.BayesianBinaryModel(
        learning_rate=0.001,
        optimizer=tf.train.AdamOptimizer,
        conv1_filters=128,
        conv1_kernel=3,
        conv2_filters=64,
        conv2_kernel=3,
        conv3_filters=64,
        conv3_kernel=3,
        dense1_units=128,
        dense1_dropout=0.5,
        log_freq=1,
        image_dim=run_config.final_size  # not initial size
    )

    run_config.train_config = train_config
    run_config.eval_config = eval_config
    run_config.model = model
    assert run_config.is_ready_to_train()

    logging.info('Parameters used: ')
    for config_object in [run_config, train_config, eval_config, model]:
        for key, value in config_object.asdict().items():
            logging.info('{}: {}'.format(key, value))
        logging.info('Next object \n')

    return run_config


def get_stratify_probs_from_csv(input_config):
    subject_df = pd.read_csv(input_config.tfrecord_loc + '.csv')
    return [1. - subject_df[input_config.label_col].mean(), subject_df[input_config.label_col].mean()]


# def train_from_disk():
#     run_config = setup()
#     run_estimator.run_estimator(run_config)


# def predict_from_disk():
#     run_config = setup()
#     return warm_start.restart_estimator(run_config)
