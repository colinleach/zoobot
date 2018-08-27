import logging
import os
import shutil
import functools
import itertools
import sqlite3
import time
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
    n_shards = (len(df) // shard_size) + 1
    df_shards = [df.iloc[n * shard_size:(n + 1) * shard_size] for n in range(n_shards)]

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


def record_acquisition_on_unlabelled(db, shard_loc, size, channels, acquisition_func):
    # iterate though the shards and get the acq. func of all unlabelled examples
    # one shard should fit in memory for one machine
    # TODO currently predicts on ALL, even labelled. Should flag labelled and exclude
    # TODO fix messy mixing of arrays, dicts, lists
    # TODO update tests with more realistic acq. function!
    subjects = read_tfrecord.load_examples_from_tfrecord(
        [shard_loc],
        read_tfrecord.matrix_id_feature_spec(size, channels)
    )
    logging.debug('Loaded subjects from {} of size {}'.format(shard_loc, size))
    # acq func expects a list of matrices
    subjects_data = [x['matrix'].reshape(size, size, channels) for x in subjects]
    acquisitions = acquisition_func(subjects_data)  # returns np array of acq. values
    acquistion_values = [acquisitions[n] for n in range(len(acquisitions))]  # convert back to list
    for subject, acquisition in zip(subjects, acquistion_values):
        if isinstance(subject['id_str'], bytes):
            subject['id_str'] = subject['id_str'].decode('utf-8')
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
        # verify that shard_loc is in at least one row of shardindex table
        cursor.execute(
            '''
            SELECT tfrecord from shardindex
            WHERE tfrecord = (:shard_loc)
            ''',
            (shard_loc,)
        )
        rows_in_shard = cursor.fetchone()
        assert rows_in_shard is not None
        # get top subjects from that shard only
        cursor.execute(  
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
    top_subjects = cursor.fetchall()  # somehow autoconverts to id_str to int...
    top_acquisitions = [str(x[0]) for x in top_subjects]  # ensure id_str really is str
    if len(top_acquisitions) == 0:
        raise ValueError
    return top_acquisitions # list of id_str's


def add_labels_to_catalog(catalog, subject_ids, subject_labels, id_col, label_col):
    # better done with a database probably
    labeller = dict(zip(subject_ids, subject_labels))
    def add_label(row):
        if row[id_col] in labeller.keys():
            return labeller[row[id_col]]
    catalog['label'] = catalog.apply(add_label, axis=1)
    return catalog


def add_top_acquisitions_to_tfrecord(catalog, db, n_subjects, shard_loc, tfrecord_loc, size):
    top_acquisitions = get_top_acquisitions(db, n_subjects, shard_loc)
    top_catalog_rows = catalog[catalog['id_str'].isin(top_acquisitions)]
    if top_catalog_rows.empty:
        raise ValueError('Fatal mismatch: top ids not found in catalog! Top: {}, Expected: {}'.format(top_acquisitions, catalog['id_str']))
    assert len(top_catalog_rows) > 0
    catalog_to_tfrecord.write_image_df_to_tfrecord(
        top_catalog_rows, 
        tfrecord_loc, 
        size, 
        columns_to_save=['id_str', 'label'], 
        append=True, # must append, don't overwrite previous training data! 
        source='fits')
    time.sleep(0.1)  # seconds


def setup(catalog, db_loc, id_col, label_col, size, shard_dir, shard_size):
    db = create_db(catalog, db_loc, id_col, label_col)
    columns_to_save = [id_col, label_col]
    # writing is slow
    write_catalog_to_tfrecord_shards(catalog, db, size, label_col, id_col, columns_to_save, shard_dir, shard_size=shard_size)


def run(catalog, db_loc, id_col, label_col, size, channels, predictor_dir, train_tfrecord_loc, train_callable, known_catalog):
    try:
        del catalog['label']  # catalog is unknown to begin with!
    except KeyError:
        pass
    
    logging.basicConfig(level=logging.INFO)
    n_samples = 20
    db = sqlite3.connect(db_loc)
    shard_locs = itertools.cycle(get_all_shard_locs(db))  # cycle through shards
    n_subjects_per_iter = 10
    max_iterations = 5

    iteration = 0
    while iteration < max_iterations:
        shard_loc = next(shard_locs)
        logging.info('Using shard_loc {}, iteration {}'.format(shard_loc, iteration))

        # train as usual, with saved_model being placed in predictor_dir
        logging.info('Training')
        train_callable()  # could be docker container run, save model

        # make predictions and save to db, could be docker container
        saved_models = os.listdir(predictor_dir)  # subfolders
        saved_models.sort(reverse=True)  # sort by name i.e. timestamp, early timestamp first
        predictor_loc = os.path.join(predictor_dir, saved_models[-1])  # the subfolder with the most recent time
        logging.info('Loading model from {}'.format(predictor_loc))
        predictor = make_predictions.load_predictor(predictor_loc)
        # inner acq. func is derived from make predictions acq func but with predictor and n_samples set
        acquisition_func = make_predictions.acquisition_func(predictor, n_samples)
        logging.info('Making and recording predictions')
        record_acquisition_on_unlabelled(db, shard_loc, size, channels, acquisition_func)

        top_acquisition_ids = get_top_acquisitions(db, n_subjects_per_iter, shard_loc=shard_loc)
        # TODO would pause here in practice

        labels = get_labels(top_acquisition_ids, known_catalog)
        
        logging.debug('Before')
        logging.debug(catalog.iloc[0][[id_col, label_col]])
        # add new labels to catalog (modify!)
        catalog = add_labels_to_catalog(catalog, top_acquisition_ids, labels, id_col, label_col)
        logging.debug('After')
        logging.debug(catalog.iloc[0][[id_col, label_col]])

        logging.info('Saving top acquisition subjects to {}, labels: {}'.format(train_tfrecord_loc, labels))
        add_top_acquisitions_to_tfrecord(catalog, db, n_subjects_per_iter, shard_loc, train_tfrecord_loc, size)

        iteration += 1

def request_labels(subject_ids):
    # TODO request labels, import from somewhere else
    pass


def recieve_labels(subject_ids, labels):
    pass


def get_labels(subject_ids, known_catalog):
    # temporary, mimic GZ
    labels = []
    for id_str in subject_ids:
        labels.append(known_catalog[known_catalog['id_str'] == id_str].squeeze())
    return labels

def get_all_shard_locs(db):
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT DISTINCT tfrecord FROM shardindex
        ORDER BY tfrecord ASC
        '''
    )
    return [row[0] for row in cursor.fetchall()]  # list of shard locs
