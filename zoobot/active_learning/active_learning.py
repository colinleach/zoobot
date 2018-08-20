import os
import shutil
import functools
import logging
import sqlite3

import tensorflow as tf
import pandas as pd
from astropy.table import Table

from zoobot.estimators import input_utils
from zoobot.tfrecord import catalog_to_tfrecord
from zoobot import shared_utilities
from zoobot.estimators import estimator_params
from zoobot.estimators import run_estimator
from zoobot.estimators import make_predictions
from zoobot.tfrecord import read_tfrecord


def create_db(db_loc):
    # primary key
    pass


def write_catalog_to_tfrecord_shards(df, db, img_size, label_col, id_col, columns_to_save, save_dir, shard_size=10000):
    assert not df.empty

    if id_col not in columns_to_save:
        columns_to_save += [id_col]

    df = df.sample(frac=1).reset_index(drop=True)  # shuffle

    # split into shards
    shard_n = 0
    n_shards = int((len(df) // shard_size) + 1)
    df_shards = [df.iloc[n * shard_size:n + 1 * shard_size] for n in range(n_shards)]

    for shard_n, df_shard in enumerate(df_shards):
        save_loc = os.path.join(save_dir, 's{}_shard_{}'.format(img_size, shard_n))
        catalog_to_tfrecord.write_image_df_to_tfrecord(df, label_col, save_loc, img_size, columns_to_save, append=False, source='fits')
        add_tfrecord_to_db(save_loc, db)
    return df


def add_tfrecord_to_db(tfrecord_loc, db, df=None):
    # scan through the record to make certain everything is truly there,
    # rather than just reading df?
    pass



def record_acquisition_on_unlabelled(db, model, shard_locs, size, channels, acqisition_func, n_samples):
    # iterate though the shards and get the acq. func of all unlabelled examples
    # shards should fit in memory for one machine
    subjects = read_tfrecord.load_examples_from_tfrecord(shard_locs, size, channels)
    predictions = make_predictions.get_samples_of_subjects(model, subjects, n_samples)
    acquisitions = acqisition_func(predictions)  # may need axis adjustment
    for subject, acquisition in zip(subjects, acquisitions):
        save_acquisition_to_db(subject, acquisition, db)


def save_acquisition_to_db(subject, acquisition, db): 
    # will overwrite previous acquisitions
    # could make faster with batches, but not needed I think
    cursor = db.cursor()  
    cursor.execute('''  
    INSERT INTO acquisitions(id, acquisition_value)
                  VALUES(:id, :acquisition_value)''',
                  {
                      'id':subject['id'], 
                      'acquisition_value':acquisition})
    db.commit()



def create_complete_tfrecord(predictions_with_catalog, params):  # predictions will be made on this tfrecord

    train_df, test_df = write_tfrecord.catalog_to_tfrecord.write_catalog_to_train_test_tfrecords(
        df=predictions_with_catalog,
        label_col='smooth-or-featured_featured-or-disk_fraction',
        train_loc=params['train_loc'],
        test_loc=params['test_loc'],
        img_size=params['img_dim'],
        columns_to_save=params['columns_to_save'],
    )
    return train_df, test_df


# currently very duplicated
def predict_on_unknown_input(unknown_tfrecord_locs, params):
    name = 'predict_on_unknown'
    return input_utils.get_input(
        tfrecord_loc=unknown_tfrecord_locs, size=params['image_dim'], name=name, batch_size=params['batch_size'], stratify=False)


def train_on_known(known_tfrecord_locs, params):
    mode = 'train_on_known'
    return input_utils.get_input(
        tfrecord_loc=known_tfrecord_locs, size=params['image_dim'], name=mode, batch_size=params['batch_size'], stratify=False)


def eval_on_known(known_tfrecord_locs, params):
    mode = 'eval_on_known'
    return input_utils.get_input(
        tfrecord_loc=known_tfrecord_locs, size=params['image_dim'], name=mode, batch_size=params['batch_size'], stratify=False)


def set_up_estimator(model_fn, params):
    # Create the Estimator
    model_fn_partial = functools.partial(model_fn, params=params)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn_partial, model_dir=params['log_dir'])
    return estimator

def run_active_learning(estimator, params, known_subjects, unknown_subjects):

    if os.path.exists(params['log_dir']):
        shutil.rmtree(params['log_dir'])

    # setup phase

    # save known subjects to tfrecord
    if not os.path.exists(params['known_tfrecord_loc']):
        catalog_to_tfrecord.write_image_df_to_tfrecord(
            df=known_subjects,
            label_col='smooth-or-featured_featured-or-disk_fraction',
            tfrecord_loc=params['known_tfrecord_loc'],
            img_size=params['img_dim'],
            columns_to_save=params['columns_to_save'],
            append=False
        )
        logging.info(
            'Saved {} catalog galaxies to {}'.format(len(known_subjects), params['known_tfrecord_loc']))

    # save unknown subjects to tfrecord
    if not os.path.exists(params['unknown_tfrecord_loc']):
        catalog_to_tfrecord.write_image_df_to_tfrecord(
            df=unknown_subjects,
            label_col='smooth-or-featured_featured-or-disk_fraction',
            tfrecord_loc=params['unknown_tfrecord_loc'],
            img_size=params['img_dim'],
            columns_to_save=params['columns_to_save'],
            append=False
        )
        logging.info(
            'Saved {} catalog galaxies to {}'.format(len(unknown_subjects), params['unknown_tfrecord_loc']))

    # initial training
    initally_known_input_partial = functools.partial(
        # TODO should be agnostic to input function and naming, except to assume images. Extend dummy!
        known_tfrecord_locs=params['known_tfrecord_loc'],
        params=params
    )

    # Set up logging for predictions
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=params['log_freq'])

    estimator.train(
        input_fn=initally_known_input_partial,
        max_steps=params['initial_train_max_steps'],
        # hooks=[logging_hook])
    )

    known_tfrecord_locs = [params['known_tfrecord_loc']]  # initially, only trained on pre-saved tfrecord. Mutable.

    unknown_input_partial = functools.partial(  # always predict on all initially-unknown subjects, slightly silly
        known_tfrecord_locs=params['unknown_tfrecord_loc'],
        params=params
    )

    # begin active learning
    epoch_n = 0
    while epoch_n < params['epochs']:
        logging.debug('Making predictions for uncertainty')
        predict_results = estimator.predict(
            input_fn=initally_known_input_partial,
            # hooks=[logging_hook],
            steps=1
        )

        #
        predictions = pd.DataFrame(predict_results)
        print(predictions)

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
    params = estimator_params.default_params()
    params.update(estimator_params.default_four_layer_architecture())
    params['img_dim'] = 64
    params['log_dir'] = 'runs/active_learning'
    params['known_tfrecord_loc'] = 'known_initially.tfrecord'
    params['initial_train_max_steps'] = 10

    columns_to_save = ['smooth-or-featured_smooth_min',
                       'smooth-or-featured_featured-or-disk_min',
                       # 'iauname', not yet supported
                       'ra',
                       'dec']
    params['columns_to_save'] = columns_to_save
    params['epochs'] = 1
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


    model_fn = run_estimator.four_layer_regression_classifier
    estimator = set_up_estimator(model_fn, params)
    run_active_learning(estimator, params, train_df)
