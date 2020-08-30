import logging
import os
import copy
import shutil
import time
from functools import partial
from typing import List

import tensorflow as tf
import pandas as pd
import matplotlib
import numpy as np

from zoobot.estimators import bayesian_estimator_funcs, input_utils, losses, efficientnet, custom_layers


class FixedEstimatorParams():
    def __init__(self, initial_size, final_size, crop_size, questions, label_cols, batch_size):
        self.initial_size=initial_size
        self.final_size = final_size
        self.crop_size = crop_size
        self.questions=questions
        self.label_cols=label_cols
        self.batch_size=batch_size

    # TODO move to shared utilities
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])


class RunEstimatorConfig():
    def __init__(
            self,
            initial_size,
            final_size,
            crop_size,
            schema: losses.Schema,
            channels=3,
            epochs=1500,  # rely on earlystopping callback
            train_steps=30,
            eval_steps=3,
            batch_size=10,
            min_epochs=0,
            patience=10,
            log_dir='runs/default_run_{}'.format(time.time()),
            save_freq=10,
            weights_loc=None,
            warm_start_settings=None
    ):  # TODO refactor for consistent order
        self.initial_size = initial_size
        self.final_size = final_size
        self.crop_size = crop_size
        self.channels = channels
        self.schema = schema
        self.epochs = epochs
        self.train_batches = train_steps
        self.eval_batches = eval_steps
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.weights_loc = weights_loc
        self.patience = patience
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


    # don't decorate, this is session creation point
    def run_estimator(self, extra_callbacks=[]):
        """
        Train and evaluate an estimator.
        `self` may well be provided by default_estimator_params.py`

        TODO save every n epochs 
        TODO enable early stopping
        TODO enable use with tf.serving
        TODO enable logging hooks?

        Args:
            self (RunEstimatorConfig): parameters controlling both estimator and train/test procedure

        Returns:
            None
        """

        logging.info('Batch {}, final size {}'.format(self.batch_size, self.final_size))
        logging.info('Train: {}'.format(self.train_config.tfrecord_loc))
        logging.info('Test: {}'.format(self.eval_config.tfrecord_loc))

        # if not self.warm_start:  # don't try to load any existing models
        #     if os.path.exists(self.log_dir):
        #         shutil.rmtree(self.log_dir)

        train_dataset = input_utils.get_input(config=self.train_config)
        test_dataset = input_utils.get_input(config=self.eval_config)

        # for im_batch, _ in train_dataset.take(1):
        #     print(im_batch.shape)
        # for im_batch, _ in test_dataset.take(1):
        #     print(im_batch.shape)
        # exit()

        checkpoint_loc = os.path.join(self.log_dir, 'in_progress')
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.log_dir, 'tensorboard'),
                histogram_freq=3,
                write_images=False,  # this actually writes the weights, terrible name
                write_graph=False,
                # profile_batch='2,10' 
                profile_batch=0   # i.e. disable profiling
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_loc,
                monitor='val_loss',
                mode='min',
                save_freq='epoch',
                save_best_only=True,
                save_weights_only=True),
            bayesian_estimator_funcs.UpdateStepCallback(
                batch_size=self.batch_size
            ),
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=self.patience),
            bayesian_estimator_funcs.UpdateStepCallback(
                batch_size=self.batch_size
            ),
            tf.keras.callbacks.TerminateOnNaN()
        ] + extra_callbacks

        # if os.path.isdir('/home/walml'):
        #     verbose=1
        # else:
        #     verbose=2
        verbose = 2
        # https://www.tensorflow.org/tensorboard/scalars_and_keras
        fit_summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'manual_summaries'))
        with fit_summary_writer.as_default():

            # for debugging
            # self.model.run_eagerly = True
            # https://www.tensorflow.org/api_docs/python/tf/keras/Model

            self.model.fit(
                train_dataset,
                validation_data=test_dataset.repeat(2),  # reduce variance from dropout, augs
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=verbose
            )

        logging.info('All epochs completed - finishing gracefully')
        # save manually outside, to avoid side-effects
        # note that the BEST model is saved as checkpoint, but self.model is the LAST model
        # to return the best model, load the last checkpoint
        logging.info('Loading and returning (best) model')
        self.model.load_weights(checkpoint_loc)  # inplace

        print(self.model.predict(test_dataset))

        print(self.model.evaluate(test_dataset))

        return self.model


# batch size changed from 256 for now, for my poor laptop
# can override less rarely specified RunEstimatorConfig defaults with **kwargs if you like
def get_run_config(initial_size, final_size, crop_size, weights_loc, log_dir, train_records, eval_records, epochs, schema, batch_size, **kwargs):

    # save these to get back the same run config across iterations? But hpefully in instructions already, or can be calculated...
    # args = {
    #     'initial_size': initial_size,
    #     'final_size': final_size,
    #     'weights_loc':
    # }
    run_config = RunEstimatorConfig(
        initial_size=initial_size,
        crop_size=crop_size,
        final_size=final_size,
        schema=schema,
        epochs=epochs,  # to tweak 2000 for overnight at 8 iters, 650 for 2h per iter
        log_dir=log_dir,
        weights_loc=weights_loc,
        batch_size=batch_size
    )

    train_config = get_train_config(train_records, schema.label_cols, run_config.batch_size, run_config.initial_size, run_config.final_size, run_config.channels)

    eval_config = get_eval_config(eval_records, schema.label_cols, run_config.batch_size, run_config.initial_size, run_config.final_size, run_config.channels)

    model = get_model(schema, run_config.initial_size, run_config.crop_size, run_config.final_size, weights_loc=weights_loc)

    run_config.assemble(train_config, eval_config, model)
    return run_config

MAX_SHIFT = 30
MAX_SHEAR = np.pi/4.
ZOOM = (1/1.65, 1/1.4)  # keras interprets zoom the other way around to normal humans, for some reason - zoom < 1 = magnification

def get_train_config(train_records, label_cols, batch_size, initial_size, final_size, channels):
    # tiny func, refactored for easy reuse
    train_config = input_utils.InputConfig(
        name='train',
        tfrecord_loc=train_records,
        label_cols=label_cols,
        stratify=False,
        shuffle=True,
        drop_remainder=True,
        repeat=False,  # Changed from True for keras, which understands to restart a dataset
        stratify_probs=None,
        geometric_augmentation=False,
        photographic_augmentation=False,
        zoom=ZOOM,
        max_shift=MAX_SHIFT,
        max_shear=MAX_SHEAR,
        contrast_range=(0.98, 1.02),
        batch_size=batch_size,
        initial_size=initial_size,
        final_size=final_size,
        channels=channels,
        greyscale=True,
        zoom_central=False  # deprecated
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
        drop_remainder=False,
        stratify_probs=None,
        geometric_augmentation=False,
        photographic_augmentation=False,
        # zoom=(2., 2.2),  # BAR MODE
        zoom=(ZOOM),  # SMOOTH MODE
        max_shift=MAX_SHIFT,
        max_shear=MAX_SHEAR,
        contrast_range=(0.98, 1.02),
        batch_size=batch_size,
        initial_size=initial_size,
        final_size=final_size,
        channels=channels,
        greyscale=True,
        zoom_central=False  # SMOOTH MODE
        # zoom_central=True  # BAR MODE
    )
    return eval_config


class CustomSequential(tf.keras.Sequential):

    def call(self, x, training):
        tf.summary.image('model_input', x, step=self.step)
        # tf.summary.image('model_input', x, step=0)
        return super().call(x, training)


def add_preprocessing_layers(model, crop_size, final_size):
    if crop_size < final_size:
        logging.warning('Crop size {} < final size {}, losing resolution'.format(crop_size, final_size))
    
    resize = True
    if np.abs(crop_size - final_size) < 10:
        logging.warning('Crop size and final size are similar: skipping resizing and cropping directly to final_size (ignoring crop_size)')
        resize = False
        crop_size = final_size

    model.add(custom_layers.PermaRandomRotation(np.pi, fill_mode='reflect'))
    model.add(custom_layers.PermaRandomFlip())
    model.add(custom_layers.PermaRandomCrop(
        crop_size, crop_size  # from 256, bad to the resize up again but need more zoom...
    ))
    if resize:
        logging.info('Using resizing, to {}'.format(final_size))
        model.add(tf.keras.layers.experimental.preprocessing.Resizing(
            final_size, final_size, interpolation='bilinear'
        ))


def get_model(schema, initial_size, crop_size, final_size, weights_loc=None):

    # dropout_rate = 0.3
    # drop_connect_rate = 0.2  # gets scaled by num blocks, 0.6ish = 1

    logging.info('Initial size {}, crop size {}, final size {}'.format(initial_size, crop_size, final_size))

    model = CustomSequential()

    model.add(tf.keras.layers.Input(shape=(initial_size, initial_size, 1)))

    add_preprocessing_layers(model, crop_size=crop_size, final_size=final_size)  # inplace

    output_dim = len(schema.label_cols)


    # model.add(bayesian_estimator_funcs.get_minst_model())  # actually this doesn't work lolz

    # now headless, with 128 neuron hidden dense layer at top
    model.add(bayesian_estimator_funcs.get_model(
        image_dim=final_size, # not initial size
        conv1_filters=32,
        conv1_kernel=3,
        conv2_filters=64,
        conv2_kernel=3,
        conv3_filters=128,
        conv3_kernel=3,
        dense1_units=128,
        dense1_dropout=0.5,
        predict_dropout=0.5,  # change this to calibrateP
        log_freq=10
    ))
    # efficientnet.custom_top_dirichlet(model, output_dim, schema)  # inplace
    # OR
    # input_shape = (final_size, final_size, 1)
    # effnet = efficientnet.EfficientNet_custom_top(
    #     schema=schema,
    #     input_shape=input_shape,
    #     get_effnet=efficientnet.EfficientNetB0
    #     # further kwargs will be passed to get_effnet
    #     # dropout_rate=dropout_rate,
    #     # drop_connect_rate=drop_connect_rate
    # )
    # model.add(effnet)

    # will be updated by callback
    model.step = tf.Variable(0, dtype=tf.int64, name='model_step', trainable=False)

    # abs_metrics = [bayesian_estimator_sequential.CustomAbsErrorByColumn(name=q.text + '_abs', start_col=start_col, end_col=end_col) for q, (start_col, end_col) in schema.named_index_groups.items()]
    # q_loss_metrics = [bayesian_estimator_sequential.CustomLossByQuestion(name=q.text + '_q_loss', start_col=start_col, end_col=end_col) for q, (start_col, end_col) in schema.named_index_groups.items()]
    # a_loss_metrics = [bayesian_estimator_sequential.CustomLossByAnswer(name=a.text + '_a_loss', col=col) for col, a in enumerate(schema.answers)]

    # this works with classic cnn, 0.03 ish. 
    # works with both sigmoid and None final activation (better with None, sigmoid not quite right here ofc)
    # With the multiq final layer, stuck on .15ish, always [[ 1.0189215 66.1367   ] (i.e. as low as possible, 0th 1-100 and 1st not mattering)
    # undoing the multiq final layer back to sigmoid i.e. (y[:, 0] - 1) / 100, works as expected
    # designed to NOT use the custom multiq final layer
    # mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    # loss = lambda x, y: mse(x[:, 0] / tf.reduce_sum(x, axis=1), y[:, 0])  # works without the multiq final layer
    # loss = lambda x, y: mse(x[:, 0] / tf.reduce_sum(x, axis=1), (y[:, 0] - 1) / 100)  # works with the multiq final layer

    # loss = lambda x, y: losses.multiquestion_loss(x, y, question_index_groups=schema.question_index_groups)

    # with convnet
    # for one question, makes standard cnn + multiq head always predict the same values [2.0518641 1.0020642]
    # for two questions, also converges to similar values, though not exactly the same values - possibly working?
    #  [1.7545501 1.0059544 1.0250697 1.2924562]
    #  [1.785771  1.00633   1.0198811 1.4103988]
    #  [2.17065   1.003661  1.0083214 1.7136166]
    # with effnet
    # for one question, predicts similar but not identical numbers, poor loss (4-5)
#     [[2.648161  1.0308543]
    #  [2.4120877 1.0264466]
    #  [2.3445587 1.0243336]
    # however, it does seem to work right for the full size version? As the concentrations are reasonable?
    # for two questions, works nicely (30 epochs)
    # multiquestion_loss = losses.get_multiquestion_loss(schema.question_index_groups)
    # loss = lambda x, y: multiquestion_loss(x, y)

    # effnet 1q. 30 epochs loss 3.0, seems to work - tho these are all pretty smooth
    # (this was with the current multiq head)
    # [[65.34233    7.150231 ]
    #  [85.74967    8.548231 ]
    #  [ 7.6572576  1.8533657]
    # convnet it does NOT seem to work with the current multiq head, gives the usual diagonal pdf - though perhaps it's just hard to get past that?
    # [[2.047638  1.0070325]
    # [2.0414152 1.0071195]
    # [2.0806828 1.0065957]
    # however, with the custom head below, it seems happy? 3.13
    import tensorflow_probability as tfp
    loss = lambda x, y: -tfp.distributions.BetaBinomial(
        tf.reduce_sum(x, axis=1), tf.nn.sigmoid(y[:, 0]) * 100, tf.nn.sigmoid(y[:, 1] * 100)).log_prob(x[:, 0])

    # loss = lambda x, y: -tfp.distributions.Binomial(tf.reduce_sum(x, axis=1), probs=(y[:, 0] - 1) / 100).log_prob(x[:, 0])

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam()
        # metrics=abs_metrics + q_loss_metrics + a_loss_metrics
    )

    # print(model)
    # exit()
    model.summary()
    # model.layers[-1].summary()

    if weights_loc:
        logging.info('Loading weights from {}'.format(weights_loc))
        load_status = model.load_weights(weights_loc)  # inplace
        # may silently fail without these
        load_status.assert_nontrivial_match()
        load_status.assert_existing_objects_matched()

    return model
