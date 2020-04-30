"""
Script version of https://colab.research.google.com/drive/1p22WgQde5ViONL8wRdONexSXL9FkZy3R#scrollTo=ylT7BWfwxRLZ
"""
import logging
import os

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # linting error
from kerastuner.tuners import Hyperband

from zoobot.estimators import input_utils, losses
from zoobot.active_learning import run_estimator_config


# need to change input_dim to n params and output_dim to n bands
def build_model(hp):

    output_dim = len(schema.answers)
    conv1_filters=32,
    conv1_kernel=1,
    conv1_activation=tf.nn.relu,
    conv2_filters=32,
    conv2_kernel=3,
    conv2_activation=tf.nn.relu,
    conv3_filters=16,
    conv3_kernel=3,
    conv3_activation=tf.nn.relu,
    dense1_units=128,
    dense1_dropout=0.5,
    dense1_activation=tf.nn.relu,
    predict_dropout=hp.Float(
        'dropout',
        min_value=0.25,
        max_value=0.75
    )


    # hp.Int('units3',
    # min_value=128,
    # max_value=1024,
    # step=64
    # )

    dropout_rate = 0  # no dropout on conv layers
    regularizer = None
    padding = 'same'
    pool1_size = 2
    pool1_strides = 2
    pool2_size = 2
    pool2_strides = 2
    pool3_size = 2
    pool3_strides = 2

    model = keras.Sequential()

    
    model.add(tf.keras.layers.Convolution2D(
            filters=conv1_filters,
            kernel_size=[conv1_kernel, conv1_kernel],
            padding=padding,
            activation=conv1_activation,
            kernel_regularizer=regularizer,
            name='model/layer1/conv1'))
    model.add(tf.keras.layers.Convolution2D(
            filters=conv1_filters,
            kernel_size=[conv1_kernel, conv1_kernel],
            padding=padding,
            activation=conv1_activation,
            kernel_regularizer=regularizer,
            name='model/layer1/conv1b'))
    model.add(tf.keras.layers.MaxPooling2D(
            pool_size=[pool1_size, pool1_size],
            strides=pool1_strides,
            name='model/layer1/pool1'))

    model.add(tf.keras.layers.Convolution2D(
            filters=conv2_filters,
            kernel_size=[conv2_kernel, conv2_kernel],
            padding=padding,
            activation=conv2_activation,
            kernel_regularizer=regularizer,
            name='model/layer2/conv2'))
    model.add(tf.keras.layers.Convolution2D(
            filters=conv2_filters,
            kernel_size=[conv2_kernel, conv2_kernel],
            padding=padding,
            activation=conv2_activation,
            kernel_regularizer=regularizer,
            name='model/layer2/conv2b'))
    model.add(tf.keras.layers.MaxPooling2D(
            pool_size=pool2_size,
            strides=pool2_strides,
            name='model/layer2/pool2'))

    model.add(tf.keras.layers.Convolution2D(
            filters=conv3_filters,
            kernel_size=[conv3_kernel, conv3_kernel],
            padding=padding,
            activation=conv3_activation,
            kernel_regularizer=regularizer,
            name='model/layer3/conv3'))
    model.add(tf.keras.layers.MaxPooling2D(
            pool_size=[pool3_size, pool3_size],
            strides=pool3_strides,
            name='model/layer3/pool3'))

    model.add(tf.keras.layers.Convolution2D(
            filters=conv3_filters,
            kernel_size=[conv3_kernel, conv3_kernel],
            padding=padding,
            activation=conv3_activation,
            kernel_regularizer=regularizer,
            name='model/layer4/conv4'))
    model.add(tf.keras.layers.MaxPooling2D(
            pool_size=[pool3_size, pool3_size],
            strides=pool3_strides,
            name='model/layer4/pool4'))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
            units=dense1_units,
            activation=dense1_activation,
            kernel_regularizer=regularizer,
            name='model/layer4/dense1'))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(
            units=output_dim,  # num outputs
            name='model/layer5/dense1')
    )

    model.add(tf.keras.layers.Dense(output_dim))  # schema passed by closure
    model.add(tf.keras.layers.Lambda(lambda x: tf.concat([tf.nn.softmax(x[:, q[0]:q[1]+1]) for q in schema.question_index_groups], axis=1)))

    model.compile(
        optimizer='adam',
        loss=lambda x, y: losses.multiquestion_loss(x, y, question_index_groups=schema.question_index_groups),
        metrics=['mean_absolute_error'])
    return model


def main(shard_dir, hyperband_iterations, max_epochs):
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    logging.info('Found GPU at: {}'.format(device_name))


    save_dir = 'temp'

    max_epochs = 1000
    patience = 50

    shard_img_size = 64
    final_size = 64
    batch_size = 16

    warm_start = False
    test = False
    train_records_dir = os.path.join(shard_dir, 'train')
    eval_records_dir = os.path.join(shard_dir, 'eval')

    train_records = [os.path.join(train_records_dir, x) for x in os.listdir(train_records_dir) if x.endswith('.tfrecord')]
    eval_records = [os.path.join(eval_records_dir, x) for x in os.listdir(eval_records_dir) if x.endswith('.tfrecord')]

    run_config = run_estimator_config.get_run_config(
        initial_size=shard_img_size,
        final_size=final_size,
        warm_start=warm_start,
        log_dir=save_dir,
        train_records=train_records,
        eval_records=eval_records,
        epochs=1,  # doesn't matter
        questions=questions,
        label_cols=label_cols,
        batch_size=batch_size
    )

    train_dataset = input_utils.get_input(config=run_config.train_config)
    test_dataset = input_utils.get_input(config=run_config.eval_config)

    tuner = Hyperband(
        build_model,
        objective='val_loss',
        hyperband_iterations=hyperband_iterations,
        max_epochs=max_epochs,
        directory='results/hyperband',
        project_name='zoobot_latest'
    )

    early_stopping = keras.callbacks.EarlyStopping(restore_best_weights=True, patience=patience)

    tuner.search(
        train_dataset,
        callbacks=[early_stopping],
        validation_data=test_dataset,
        batch_size=batch_size
    )

    tuner.results_summary()

    models = tuner.get_best_models(num_models=5)

    for n, model in enumerate(models):
        logging.info(f'Model {n}')
        logging.info(model.summary())

if __name__ == '__main__':


    tf.config.optimizer.set_jit(True)  # XLA compilation for keras model


    questions = [
        'smooth-or-featured',
        'has-spiral-arms',
        'bar',
        'bulge-size'
    ]
    label_cols = [
        'smooth-or-featured_smooth',
        'smooth-or-featured_featured-or-disk',
        'has-spiral-arms_yes',
        'has-spiral-arms_no',
        'bar_strong',
        'bar_weak',
        'bar_no',
        'bulge-size_dominant',
        'bulge-size_large',
        'bulge-size_moderate',
        'bulge-size_small',
        'bulge-size_none'
    ]
    schema = losses.Schema(label_cols, questions)

    shard_dir = '/home/walml/repos/zoobot/data/decals/shards/multilabel_master_filtered_128'

    hyperband_iterations = 5
    max_epochs = 1000
    main(shard_dir, hyperband_iterations, max_epochs)
