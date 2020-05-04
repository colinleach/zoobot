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

from zoobot.estimators import input_utils, losses, efficientnet
from zoobot.active_learning import run_estimator_config


# need to change input_dim to n params and output_dim to n bands
def build_conv_model(hp=None):

    output_dim = len(schema.answers)
    conv1_filters=32
    conv1_kernel=1
    conv2_filters=32
    conv2_kernel=3
    conv3_filters=16
    conv3_kernel=3
    dense1_units=128

    if hp is None:
        dropout_rate = 0.5
    else:
        dropout_rate=hp.Float(  # no dropout on conv layers
            'dropout',
            min_value=0.25,
            max_value=0.75
        )
 
    regularizer = None
    padding = 'same'
    pool1_size = 2
    pool1_strides = 2
    pool2_size = 2
    pool2_strides = 2
    pool3_size = 2
    pool3_strides = 2

    model = keras.Sequential()

    
    # model.add(tf.keras.layers.Conv2D(
    #         filters=conv1_filters,
    #         kernel_size=[conv1_kernel, conv1_kernel],
    #         padding=padding,
    #         activation=tf.nn.relu,
    #         kernel_regularizer=regularizer,
    #         name='model/layer1/conv1'))
    # model.add(tf.keras.layers.Conv2D(
    #         filters=conv1_filters,
    #         kernel_size=[conv1_kernel, conv1_kernel],
    #         padding=padding,
    #         activation=tf.nn.relu,
    #         kernel_regularizer=regularizer,
    #         name='model/layer1/conv1b'))
    # model.add(tf.keras.layers.MaxPooling2D(
    #         pool_size=[pool1_size, pool1_size],
    #         strides=pool1_strides,
    #         name='model/layer1/pool1'))

    # model.add(tf.keras.layers.Conv2D(
    #         filters=conv2_filters,
    #         kernel_size=[conv2_kernel, conv2_kernel],
    #         padding=padding,
    #         activation=tf.nn.relu,
    #         kernel_regularizer=regularizer,
    #         name='model/layer2/conv2'))
    # model.add(tf.keras.layers.Conv2D(
    #         filters=conv2_filters,
    #         kernel_size=[conv2_kernel, conv2_kernel],
    #         padding=padding,
    #         activation=tf.nn.relu,
    #         kernel_regularizer=regularizer,
    #         name='model/layer2/conv2b'))
    # model.add(tf.keras.layers.MaxPooling2D(
    #         pool_size=pool2_size,
    #         strides=pool2_strides,
    #         name='model/layer2/pool2'))

    # model.add(tf.keras.layers.Conv2D(
    #         filters=conv3_filters,
    #         kernel_size=[conv3_kernel, conv3_kernel],
    #         padding=padding,
    #         activation=tf.nn.relu,
    #         kernel_regularizer=regularizer,
    #         name='model/layer3/conv3'))
    # model.add(tf.keras.layers.MaxPooling2D(
    #         pool_size=[pool3_size, pool3_size],
    #         strides=pool3_strides,
    #         name='model/layer3/pool3'))

    # model.add(tf.keras.layers.Conv2D(
    #         filters=conv3_filters,
    #         kernel_size=[conv3_kernel, conv3_kernel],
    #         padding=padding,
    #         activation=tf.nn.relu,
    #         kernel_regularizer=regularizer,
    #         name='model/layer4/conv4'))
    # model.add(tf.keras.layers.MaxPooling2D(
    #         pool_size=[pool3_size, pool3_size],
    #         strides=pool3_strides,
    #         name='model/layer4/pool4'))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
            units=dense1_units,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='model/layer4/dense1'))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(output_dim))  # schema passed by closure
    model.add(tf.keras.layers.Lambda(lambda x: tf.concat([tf.nn.softmax(x[:, q[0]:q[1]+1]) for q in schema.question_index_groups], axis=1)))

    model.compile(
        optimizer='adam',
        loss=lambda x, y: losses.multiquestion_loss(x, y, question_index_groups=schema.question_index_groups),
        metrics=['mean_absolute_error'])
    return model

def main(shard_dir, hyperband_iterations, max_epochs, schema):


    def build_efficientnet(hp=None):
            # https://keras-team.github.io/keras-tuner/documentation/hyperparameters/#float-method

            #     flops = depth_coefficient * width_coefficient ** 2 * resolution ** 2

        if hp is None:
            depth_coefficient = 1.
            width_coefficient = 1.
            dropout_rate = 0.2
            drop_connect_rate = 0.2
        else:

            # resolution = 64  now passed by closure
            depth_coefficient = hp.Float(min_value=0.5, max_value=2.0, step=0.2, name='depth_coefficient')
            width_coefficient = 1. / np.sqrt(depth_coefficient)

        #     resolution = hp.Int(128, 350, step=32)
        #     width_coefficient hp.Float(0.5, 2.0, step=0.2)
        #     depth_coefficient = 1 / (np.sqrt(width_coefficient) * np.sqrt(depth_coefficient))

            dropout_rate = hp.Float(min_value=0.1, max_value=0.4, step=0.1, name='dropout_rate')
            drop_connect_rate = hp.Float(min_value=0.1, max_value=0.4, step=0.1, name='drop_connect_rate')


        return efficientnet.EfficientNet_custom_top(
            schema,  # by closure
            input_shape=(resolution, resolution, 1),
            batch_size=batch_size,
            get_effnet=efficientnet.EfficientNet,
            # remaining args passed through
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            dropout_rate=dropout_rate,
            drop_connect_rate=drop_connect_rate,
            default_resolution=resolution,
        )

    

    # doing this messes up CUDA gpu growth fix, don't!
    # device_name = tf.test.gpu_device_name()
    # if device_name != '/device:GPU:0':
    #     raise SystemError('GPU device not found')
    # logging.info('Found GPU at: {}'.format(device_name))


    save_dir = 'temp'

    max_epochs = 1000
    patience = 50

    shard_img_size = 64
    resolution = 64
    batch_size = 6

    warm_start = False
    train_records_dir = os.path.join(shard_dir, 'train')
    eval_records_dir = os.path.join(shard_dir, 'eval')

    train_records = [os.path.join(train_records_dir, x) for x in os.listdir(train_records_dir) if x.endswith('.tfrecord')]
    eval_records = [os.path.join(eval_records_dir, x) for x in os.listdir(eval_records_dir) if x.endswith('.tfrecord')]

    run_config = run_estimator_config.get_run_config(
        initial_size=shard_img_size,
        final_size=resolution,
        warm_start=warm_start,
        log_dir=save_dir,
        train_records=train_records,
        eval_records=eval_records,
        epochs=1,  # doesn't matter
        questions=questions,
        schema=schema,
        batch_size=batch_size
    )

    train_dataset = input_utils.get_input(config=run_config.train_config)
    test_dataset = input_utils.get_input(config=run_config.eval_config)

    print(run_config.train_config.batch_size)
    for x, y in train_dataset.take(5):
        print(x.shape)
    exit()

    early_stopping = keras.callbacks.EarlyStopping(restore_best_weights=True, patience=patience)

    # model = build_conv_model(hp=None)
    # model.fit(
    #     train_dataset,
    #     callbacks=[early_stopping],
    #     validation_data=test_dataset
    # )

    model = build_efficientnet(hp=None)
    model.run_eagerly = True
    model.fit(
        train_dataset,
        callbacks=[early_stopping],
        validation_data=test_dataset
    )

    exit()

    tuner = Hyperband(
        build_efficientnet,
        objective='val_loss',
        hyperband_iterations=hyperband_iterations,
        max_epochs=max_epochs,
        directory='results/hyperband',
        project_name='zoobot_latest'
    )

    tuner.search(
        train_dataset,
        callbacks=[early_stopping],
        validation_data=test_dataset,
        # batch_size=batch_size
    )

    tuner.results_summary()

    models = tuner.get_best_models(num_models=5)

    for n, model in enumerate(models):
        logging.info(f'Model {n}')
        logging.info(model.summary())

if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

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
    schema = losses.Schema(label_cols, questions, version='decals')

    if os.path.isdir('/home/walml'):
        base_dir = '/home/walml/'
    else:
        base_dir = os.environ['DATA']

    # optimising on all decals galaxies (made w/ make_decals_tfrecords, not sim shards)
    shard_dir = os.path.join(base_dir, 'repos/zoobot/data/decals/shards/multilabel_master_filtered_128')

    hyperband_iterations = 5
    max_epochs = 1000
    main(shard_dir, hyperband_iterations, max_epochs, schema)
