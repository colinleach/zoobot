import os
import shutil
import functools
import logging
import sqlite3
from collections import namedtuple

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


def create_db(catalog, db_loc, id_col, label_col):  # very similar to empty db fixture
    
    db = sqlite3.connect(db_loc)

    cursor = db.cursor()

    cursor.execute(
        '''
        CREATE TABLE catalog(
            id_str STRING PRIMARY KEY,
            label INT)
        '''
    )
    db.commit()

    cursor.execute(
        '''
        CREATE TABLE shardindex(
            id_str STRING PRIMARY KEY,
            tfrecord TEXT)
        '''
    )
    db.commit()

    cursor.execute(
        '''
        CREATE TABLE acquisitions(
            id_str STRING PRIMARY KEY,
            acquisition_value FLOAT)
        '''
    )
    db.commit()

    add_catalog_to_db(catalog, db, id_col, label_col)

    return db


def write_catalog_to_tfrecord_shards(df, db, img_size, label_col, id_col, columns_to_save, save_dir, shard_size=10000):
    assert not df.empty
    assert id_col in columns_to_save
    assert label_col in columns_to_save

    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    # split into shards
    shard_n = 0
    n_shards = int((len(df) // shard_size) + 1)
    df_shards = [df.iloc[n * shard_size:n + 1 * shard_size] for n in range(n_shards)]

    for shard_n, df_shard in enumerate(df_shards):
        save_loc = os.path.join(save_dir, 's{}_shard_{}'.format(img_size, shard_n))
        catalog_to_tfrecord.write_image_df_to_tfrecord(df_shard, save_loc, img_size, columns_to_save, append=False, source='fits')
        add_tfrecord_to_db(save_loc, db, df_shard, id_col)
    return df


def add_catalog_to_db(df, db, id_col, label_col):
    catalog_entries = list(df[[id_col, label_col]].itertuples(index=False, name='CatalogEntry'))
    cursor = db.cursor()
    cursor.executemany(
        '''
        INSERT INTO catalog(id_str, label) VALUES(?,?)''',
        catalog_entries
    )
    db.commit()


def add_tfrecord_to_db(tfrecord_loc, db, df, id_col):
    # scan through the record to make certain everything is truly there,
    # rather than just reading df?
    # eventually, consider the catalog being SQL as source-of-truth and csv output
    # ShardIndexEntry = namedtuple('ShardIndexEntry', 'id, tfrecord')
    shard_index_entries = list(zip(
        df[id_col].values, 
        [tfrecord_loc for n in range(len(df))]
    ))
    # shard_index_entries = list(map(
    #     lambda x, y: ShardIndexEntry._make(id=x, tfrecord=y),
    #     d,
          # could alternatively modify df beforehand
    # ))  
    cursor = db.cursor()
    cursor.executemany(
        '''
        INSERT INTO shardindex(id_str, tfrecord) VALUES(?,?)''',
        shard_index_entries
    )
    db.commit()


def record_acquisition_on_unlabelled(db, shard_locs, size, channels, acquisition_func):
    # iterate though the shards and get the acq. func of all unlabelled examples
    # shards should fit in memory for one machine
    # TODO currently predicts on ALL, even labelled. Should flag labelled and exclude
    subjects = read_tfrecord.load_examples_from_tfrecord(shard_locs, read_tfrecord.matrix_id_feature_spec(size, channels))
    acquisitions = acquisition_func(subjects)
    for subject, acquisition in zip(subjects, acquisitions):
        save_acquisition_to_db(subject, acquisition, db)


def save_acquisition_to_db(subject, acquisition, db): 
    # will overwrite previous acquisitions
    # could make faster with batches, but not needed I think
    # TODO needs test for duplicate values/replace behaviour
    cursor = db.cursor()  
    cursor.execute('''  
    INSERT OR REPLACE INTO acquisitions(id_str, acquisition_value)  
                  VALUES(:id_str, :acquisition_value)''',
                  {
                      'id_str':subject['id_str'], 
                      'acquisition_value':acquisition})
    db.commit()


def get_top_acquisitions(db, n_subjects=1000, shard_loc=None):
    cursor = db.cursor()
    if shard_loc is None:  # top subjects from any shard
        cursor.execute(
            '''
            SELECT id_str, acquisition_value 
            FROM acquisitions 
            ORDER BY acquisition_value DESC
            LIMIT (:n_subjects)
            ''',
            (n_subjects,)
        )
    else:
        cursor.execute(  # top subjects from that shard only
            '''
            SELECT acquisitions.id_str, acquisitions.acquisition_value, shardindex.id_str, shardindex.tfrecord
            FROM acquisitions 
            INNER JOIN shardindex ON acquisitions.id_str = shardindex.id_str
            WHERE shardindex.tfrecord = (:shard_loc)
            ORDER BY acquisition_value DESC
            LIMIT (:n_subjects)
            ''',
            (shard_loc, n_subjects),
        )
    top_subjects = cursor.fetchall()
    return [x[0] for x in top_subjects]  # list of id_str's


def add_top_acquisitions_to_tfrecord(catalog, db, n_subjects, shard_loc, tfrecord_loc, size):
    top_acquisitions = get_top_acquisitions(db, n_subjects, shard_loc)
    assert len(top_acquisitions) > 0
    top_catalog_rows = catalog[catalog['id_str'].isin(top_acquisitions)]
    assert len(top_catalog_rows) > 0
    catalog_to_tfrecord.write_image_df_to_tfrecord(
        top_catalog_rows, 
        tfrecord_loc, 
        size, 
        columns_to_save=['id_str', 'label'], 
        append=True, # must append, don't overwrite previous training data! 
        source='fits')


def setup(catalog, db_loc, id_col, label_col, size, save_dir):
    db = create_db(catalog, db_loc, id_col, label_col)
    columns_to_save = [id_col, label_col]
    # writing is slow
    write_catalog_to_tfrecord_shards(catalog, db, size, label_col, id_col, columns_to_save, save_dir, shard_size=10000)


def run(catalog, db_loc, id_col, label_col, size, channels, predictor_dir, train_tfrecord_loc):
    n_samples = 20
    db = sqlite3.connect(db_loc)
    shard_locs = get_all_shard_locs(db)
    n_subjects_per_iter = 1000
    max_iterations = 10

    predictor_loc = os.listdir(predictor_dir).sort()[-1]  # the subfolder with the most recent time
    iteration = 0
    while iteration < max_iterations:
        for shard_loc in shard_locs:
            predictor = make_predictions.load_predictor(predictor_loc)
            acquisition_func = make_predictions.acquisition_func(predictor, n_samples)
            record_acquisition_on_unlabelled(db, [shard_loc], size, channels, acquisition_func)
            add_top_acquisitions_to_tfrecord(catalog, db, n_subjects_per_iter, shard_loc, train_tfrecord_loc, size)
            # TODO train as usual, with saved_model being placed in predictor_dir
            iteration += 1


def get_all_shard_locs(db):
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT tfrecord FROM shardindex
        '''
    )
    return [row[0] for row in cursor.fetchall()]  # list of shard locs


if __name__ == '__main__':

    logging.basicConfig(
        filename='active_learning.log',
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.INFO)

    predictions_with_catalog_loc = '/data/repos/galaxy-zoo-panoptes/reduction/data/output/panoptes_predictions_with_catalog.csv'
    predictions_with_catalog = pd.read_csv(predictions_with_catalog_loc)
    logging.info('Loaded {} catalog galaxies with predictions'.format(len(predictions_with_catalog)))
