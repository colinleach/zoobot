import logging
import os

import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from zoobot.estimators import bayesian_estimator_funcs, run_estimator, input_utils, losses


def get_run_config(params, log_dir, train_records, eval_records, learning_rate, epochs, label_cols, train_steps=15, eval_steps=5, batch_size=256, min_epochs=2000, early_stopping_window=10, max_sadness=5., save_freq=10):
    # TODO enforce keyword only arguments
    channels = 3

    run_config = run_estimator.RunEstimatorConfig(
        initial_size=params.initial_size,
        final_size=params.final_size,
        channels=channels,
        label_cols=label_cols,
        epochs=epochs,  # to tweak 2000 for overnight at 8 iters, 650 for 2h per iter
        train_steps=train_steps,  # compensating for doubling the batch, still want to measure often
        eval_steps=eval_steps,
        batch_size=batch_size,  # increased from 128 for less training noise
        min_epochs=min_epochs,  # no early stopping, just run it overnight
        early_stopping_window=early_stopping_window,  # to tweak
        max_sadness=max_sadness,  # to tweak
        log_dir=log_dir,
        save_freq=save_freq,
        warm_start=params.warm_start
    )

    train_config = input_utils.InputConfig(
        name='train',
        tfrecord_loc=train_records,
        label_cols=run_config.label_cols,
        stratify=False,
        shuffle=True,
        repeat=True,
        stratify_probs=None,
        geometric_augmentation=True,
        photographic_augmentation=True,
        # zoom=(2., 2.2),  # BAR MODE
        zoom=(1.1, 1.3),  # SMOOTH MODE
        contrast_range=(0.98, 1.02),
        fill_mode='wrap',
        batch_size=run_config.batch_size,
        initial_size=run_config.initial_size,
        final_size=run_config.final_size,
        channels=run_config.channels,
        greyscale=True,
        zoom_central=False  # SMOOTH MODE
        # zoom_central=True  # BAR MODE
    )

    eval_config = input_utils.InputConfig(
        name='eval',
        tfrecord_loc=eval_records,
        label_cols=run_config.label_cols,
        stratify=False,
        shuffle=True,
        repeat=False,
        stratify_probs=None,
        geometric_augmentation=True,
        photographic_augmentation=True,
        # zoom=(2., 2.2),  # BAR MODE
        zoom=(1.1, 1.3),  # SMOOTH MODE
        contrast_range=(0.98, 1.02),
        fill_mode='wrap',
        batch_size=run_config.batch_size,
        initial_size=run_config.initial_size,
        final_size=run_config.final_size,
        channels=run_config.channels,
        greyscale=True,
        zoom_central=False  # SMOOTH MODE
        # zoom_central=True  # BAR MODE
    )

    # schema = losses.get_schema_from_label_cols(label_cols=run_config.label_cols, questions=['smooth', 'spiral'])
    questions = ['smooth', 'spiral']
    question_indices = losses.get_indices_from_label_cols(label_cols=run_config.label_cols, questions=questions)
    model = bayesian_estimator_funcs.BayesianModel(
        output_dim=len(run_config.label_cols),
        learning_rate=learning_rate,
        optimizer=tf.train.AdamOptimizer,
        conv1_filters=32,
        conv1_kernel=3,
        conv2_filters=64,
        conv2_kernel=3,
        conv3_filters=128,
        conv3_kernel=3,
        dense1_units=128,
        dense1_dropout=0.5,
        predict_dropout=0.5,  # change this to calibrate
        regression=True,  # important!
        log_freq=10,
        image_dim=run_config.final_size,  # not initial size
        calculate_loss = lambda x, y: losses.multiquestion_loss(x, y, question_index_groups=question_indices, num_questions=len(questions), dtype=tf.int32)
        # calculate_loss=lambda x, y: losses.multinomial_loss(x, y, output_dim=len(run_config.label_cols))  # assumes labels are columns of successes and predictions are cols of prob.
    )  # WARNING will need to be updated for multiquestion

    run_config.assemble(train_config, eval_config, model)
    return run_config
