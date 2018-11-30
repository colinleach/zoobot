import logging
import os

import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from zoobot.estimators import bayesian_estimator_funcs, run_estimator, input_utils, warm_start


def get_run_config(active_config):

    channels = 3

    run_config = run_estimator.RunEstimatorConfig(
        initial_size=active_config.shards.initial_size,
        final_size=active_config.shards.final_size,
        channels=channels,
        label_col='label',
        epochs=265,  # as temporary test
        train_steps=30,
        eval_steps=3,
        batch_size=128,
        min_epochs=265,  # don't stop early automatically
        early_stopping_window=10,
        max_sadness=4.,
        log_dir=active_config.estimator_dir,
        save_freq=10,
        warm_start=False  # Will restore previous run from disk, if saved
    )

    train_config = input_utils.InputConfig(
        name='train',
        tfrecord_loc=active_config.shards.train_tfrecord_loc,
        label_col=run_config.label_col,
        stratify=True,
        shuffle=True,
        repeat=True,
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
    train_config.set_stratify_probs_from_csv(train_config.tfrecord_loc + '.csv')

    eval_config = input_utils.InputConfig(
        name='eval',
        tfrecord_loc=active_config.shards.eval_tfrecord_loc,
        label_col=run_config.label_col,
        stratify=True,
        shuffle=True,
        repeat=False,
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
    eval_config.set_stratify_probs_from_csv(train_config.tfrecord_loc + '.csv')  # eval not allowed

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

    run_config.assemble(train_config, eval_config, model)
    return run_config
