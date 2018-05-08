import os
import shutil
import functools
import logging

import tensorflow as tf
import pandas as pd
from astropy.table import Table

from zoobot.estimators import input_utils
from zoobot.tfrecord import gz2_to_tfrecord
from zoobot import shared_utilities
from zoobot.estimators import architecture_values
from zoobot.estimators import run_estimator


def create_complete_tfrecord(predictions_with_catalog, params):  # predictions will be made on this tfrecord

    train_df, test_df = gz2_to_tfrecord.write_catalog_to_train_test_tfrecords(
        df=predictions_with_catalog,
        label_col='smooth-or-featured_featured-or-disk_fraction',
        train_loc=params['train_loc'],
        test_loc=params['test_loc'],
        img_size=params['img_dim'],
        columns_to_save=params['columns_to_save'],
    )
    return train_df, test_df


def predict_input(params):  # deterministic
    name = 'predict'
    # return input_utils.input(
    #     tfrecord_loc=params['train_loc'], size=params['image_dim'], name=name, batch=params['batch_size'], stratify=params['eval_stratify'])
    return input_utils.input(
        tfrecord_loc=params['active_tfrecord_loc'], size=params['image_dim'], name=name, batch=params['batch_size'], stratify=params['train_stratify'])


def active_train_input(params):
    mode = 'active_training'
    return input_utils.input(
        tfrecord_loc=params['active_tfrecord_loc'], size=params['image_dim'], name=mode, batch=params['batch_size'], stratify=params['train_stratify'])


def eval_input(params):
    mode = 'test'
    return input_utils.input(
        tfrecord_loc=params['test_loc'], size=params['image_dim'], name=mode, batch=params['batch_size'], stratify=params['eval_stratify'])


def set_up_estimator(model_fn, params):
    # Create the Estimator
    model_fn_partial = functools.partial(model_fn, params=params)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn_partial, model_dir=params['log_dir'])
    return estimator


def run_active_learning(estimator, params, catalog):

    if os.path.exists(params['log_dir']):
        shutil.rmtree(params['log_dir'])

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=params['log_freq'])

    predict_input_partial = functools.partial(predict_input, params=params)
    active_train_input_partial = functools.partial(active_train_input, params=params)
    eval_input_partial = functools.partial(eval_input, params=params)

    estimator.train(
        input_fn=active_train_input_partial,
        steps=1,
        max_steps=params['max_train_batches'],
        hooks=[logging_hook])

    epoch_n = 0
    params['epochs'] = 1
    while epoch_n < params['epochs']:

        logging.debug('Making predictions for uncertainty')
        predict_results = estimator.predict(
            input_fn=active_train_input_partial,
            hooks=[logging_hook]
        )
        print('Got results: {}'.format(predict_results))

        for result in predict_results:
            print('result: {}'.format(result))

        # predict_results = tf.Print(predict_results, [predict_results])
        #
        # output = tf.multiply(predict_results, 1)

        #
        # uncertain_galaxies = catalog
        #
        # logging.debug('Appending least confident subjects to train tfrecord')
        # gz2_to_tfrecord.write_image_df_to_tfrecord(
        #     df=uncertain_galaxies,
        #     label_col='smooth-or-featured_featured-or-disk_fraction',
        #     tfrecord_loc=params['active_tfrecord_loc'],
        #     img_size=params['img_dim'],
        #     columns_to_save=params['columns_to_save'],
        #     append=True
        # )
        # logging.info('Saved {} catalog galaxies to {}'.format(len(predictions_with_catalog), params['tfrecord_loc']))
        #
        # logging.debug('Training')
        # # Train the estimator
        # estimator.train(
        #     input_fn=active_train_input_partial,
        #     steps=params['train_batches'],
        #     max_steps=params['max_train_batches'],
        #     hooks=[logging_hook]
        # )
        # # Evaluate the estimator and print results
        # print('eval begins')
        # eval_results = estimator.evaluate(
        #     input_fn=eval_input_partial,
        #     steps=params['eval_batches'],
        #     hooks=[logging_hook]
        # )
        # print(eval_results)
        #
        # print('saving model at epoch {}'.format(epoch_n))
        # Not yet implemented
        # estimator.export_savedmodel(
        #     export_dir_base="/path/to/model",
        #     serving_input_receiver_fn=serving_input_receiver_fn)

        epoch_n += 1


def get_active_learning_params():
    params = architecture_values.default_params()
    params.update(architecture_values.default_four_layer_architecture())
    params['img_dim'] = 64
    params['log_dir'] = 'runs/active_learning'
    params['train_loc'] = 'all_panoptes_galaxies_train.tfrecord'
    params['test_loc'] = 'all_panoptes_galaxies_test.tfrecord'
    params['active_tfrecord_loc'] = 'active.tfrecord'

    columns_to_save = ['smooth-or-featured_smooth_min',
                       'smooth-or-featured_featured-or-disk_min',
                       # 'iauname', not yet supported
                       'ra',
                       'dec']
    params['columns_to_save'] = columns_to_save
    return params


if __name__ == '__main__':

    logging.basicConfig(
        filename='active_learning.log',
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.INFO)

    predictions_with_catalog_loc = '/data/repos/galaxy-zoo-panoptes/reduction/data/output/panoptes_predictions_with_catalog.csv'
    predictions_with_catalog = pd.read_csv(predictions_with_catalog_loc)
    logging.info('Loaded {} catalog galaxies with predictions'.format(len(predictions_with_catalog)))

    params = get_active_learning_params()

    # train_df, test_df = create_complete_tfrecord(predictions_with_catalog, params)  # save train and test tfrecords
    train_df = predictions_with_catalog[:3159]  # for now, just pick first 3k

    gz2_to_tfrecord.write_image_df_to_tfrecord(
        df=train_df[:100],  # save starter active learning tfrecord
        label_col='smooth-or-featured_featured-or-disk_fraction',
        tfrecord_loc=params['active_tfrecord_loc'],
        img_size=params['img_dim'],
        columns_to_save=params['columns_to_save'],
        append=False  # overwrite any existing active learning tfrecord
    )
    logging.info('Saved {} catalog galaxies to {}'.format(len(predictions_with_catalog), params['active_tfrecord_loc']))

    model_fn = run_estimator.four_layer_regression_classifier
    estimator = set_up_estimator(model_fn, params)
    run_active_learning(estimator, params, train_df)
