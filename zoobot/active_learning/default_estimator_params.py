import logging
import os
import time
from typing import List

import tensorflow as tf
import pandas as pd
import matplotlib
# 

from zoobot.estimators import bayesian_estimator_funcs, input_utils, losses

class RunEstimatorConfig():

    def __init__(
            self,
            initial_size,
            final_size,
            channels,
            label_cols: List,
            epochs=50,
            train_steps=30,
            eval_steps=3,
            batch_size=128,
            min_epochs=0,
            early_stopping_window=10,
            max_sadness=4.,
            log_dir='runs/default_run_{}'.format(time.time()),
            save_freq=10,
            warm_start=True,
            warm_start_settings=None
    ):  # TODO refactor for consistent order
        self.initial_size = initial_size
        self.final_size = final_size
        self.channels = channels
        self.label_cols = label_cols
        self.epochs = epochs
        self.train_batches = train_steps
        self.eval_batches = eval_steps
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.warm_start = warm_start
        self.max_sadness = max_sadness
        self.early_stopping_window = early_stopping_window
        self.min_epochs = min_epochs
        self.train_config = None
        self.eval_config = None
        self.model = None
        self.warm_start_settings = warm_start_settings

    
    def assemble(self, train_config, eval_config, model):
        self.train_config = train_config
        self.eval_config = eval_config
        self.model = model
        assert self.is_ready_to_train()

    def is_ready_to_train(self):
        # TODO can make this check much more comprehensive
        return (self.train_config is not None) and (self.eval_config is not None)

    def log(self):
        logging.info('Parameters used: ')
        for config_object in [self, self.train_config, self.eval_config, self.model]:
            for key, value in config_object.asdict().items():
                logging.info('{}: {}'.format(key, value))

    # TODO move to shared utilities
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])


def get_run_config(params, log_dir, train_records, eval_records, learning_rate, epochs, label_cols, questions, train_steps=15, eval_steps=5, batch_size=256, min_epochs=2000, early_stopping_window=10, max_sadness=5., save_freq=10):
    # TODO enforce keyword only arguments
    channels = 3

    run_config = RunEstimatorConfig(
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

    train_config = get_train_config(train_records, run_config.label_cols, run_config.batch_size, run_config.initial_size, run_config.final_size, run_config.channels)

    eval_config = get_eval_config(eval_records, run_config.label_cols, run_config.batch_size, run_config.initial_size, run_config.final_size, run_config.channels)

    model = get_model(label_cols, questions, run_config.final_size)

    run_config.assemble(train_config, eval_config, model)
    return run_config


def get_train_config(train_records, label_cols, batch_size, initial_size, final_size, channels):
    # tiny func, refactored for easy reuse
    train_config = input_utils.InputConfig(
        name='train',
        tfrecord_loc=train_records,
        label_cols=label_cols,
        stratify=False,
        shuffle=False,  # temporarily turned off due to shuffle op error
        repeat=False,  # Changed from True for keras, which understands to restart a dataset
        stratify_probs=None,
        geometric_augmentation=True,
        photographic_augmentation=True,
        # zoom=(2., 2.2),  # BAR MODE
        zoom=(1.1, 1.3),  # SMOOTH MODE
        contrast_range=(0.98, 1.02),
        fill_mode='wrap',
        batch_size=batch_size,
        initial_size=initial_size,
        final_size=final_size,
        channels=channels,
        greyscale=True,
        zoom_central=False  # SMOOTH MODE
        # zoom_central=True  # BAR MODE
    )
    return train_config


def get_eval_config(eval_records, label_cols, batch_size, initial_size, final_size, channels):
    # tiny func, refactored for easy reuse
    eval_config = input_utils.InputConfig(
        name='eval',
        tfrecord_loc=eval_records,
        label_cols=label_cols,
        stratify=False,
        shuffle=False,  # see above
        repeat=False,
        stratify_probs=None,
        geometric_augmentation=True,
        photographic_augmentation=True,
        # zoom=(2., 2.2),  # BAR MODE
        zoom=(1.1, 1.3),  # SMOOTH MODE
        contrast_range=(0.98, 1.02),
        fill_mode='wrap',
        batch_size=batch_size,
        initial_size=initial_size,
        final_size=final_size,
        channels=channels,
        greyscale=True,
        zoom_central=False  # SMOOTH MODE
        # zoom_central=True  # BAR MODE
    )
    return eval_config

def get_model(label_cols, questions, final_size):
    schema = losses.Schema(label_cols, questions)
    model = bayesian_estimator_funcs.BayesianModel(
        image_dim=final_size, # not initial size
        output_dim=len(label_cols),  # will predict all label columns, in this order
        schema=schema,
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
        log_freq=10
    )  # WARNING will need to be updated for multiquestion

    model.compile(
        loss=lambda x, y: losses.multiquestion_loss(x, y, question_index_groups=schema.question_index_groups),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[bayesian_estimator_funcs.CustomMSEByColumn(name=q.text, start_col=start_col, end_col=end_col) for q, (start_col, end_col) in schema.named_index_groups.items()]
    )
    return model
