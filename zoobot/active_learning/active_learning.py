import logging
import os
import shutil
import functools
import itertools
import sqlite3
import time
from collections import namedtuple

import pytest  # nice assertions
import tensorflow as tf
import pandas as pd
import numpy as np

from zoobot.estimators import input_utils
from zoobot.tfrecord import catalog_to_tfrecord, create_tfrecord
from zoobot import shared_utilities
from zoobot.estimators import estimator_params
from zoobot.estimators import run_estimator
from zoobot.estimators import make_predictions
from zoobot.tfrecord import read_tfrecord

from zoobot.active_learning import mock_panoptes


def create_db(catalog, db_loc, id_col):  # very similar to empty db fixture
    
    db = sqlite3.connect(db_loc)

    cursor = db.cursor()

    cursor.execute(
        '''
        CREATE TABLE catalog(
            id_str STRING PRIMARY KEY,
            label INT DEFAULT NULL,
            fits_loc STRING)
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

    add_catalog_to_db(catalog, db, id_col)  # does NOT add labels, labels are unknown at this point

    return db


def write_catalog_to_tfrecord_shards(df, db, img_size, id_col, columns_to_save, save_dir, shard_size=10000):
    assert not df.empty
    assert id_col in columns_to_save

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


def add_catalog_to_db(df, db, id_col):
    catalog_entries = list(df[[id_col, 'fits_loc']].itertuples(index=False, name='CatalogEntry'))
    cursor = db.cursor()
    cursor.executemany(
        '''
        INSERT INTO catalog(id_str, label, fits_loc) VALUES(?,NULL,?)''',
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
            SELECT acquisitions.id_str, acquisitions.acquisition_value
            FROM acquisitions
            INNER JOIN catalog ON acquisitions.id_str = catalog.id_str
            WHERE catalog.label IS NULL
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
        # get top subjects from that shard only TODO catalog.id_str is null
        cursor.execute(  
            '''
            SELECT acquisitions.id_str, acquisitions.acquisition_value, shardindex.id_str, shardindex.tfrecord
            FROM acquisitions 
            INNER JOIN shardindex ON acquisitions.id_str = shardindex.id_str
            INNER JOIN catalog ON acquisitions.id_str = catalog.id_str
            WHERE shardindex.tfrecord = (:shard_loc) AND catalog.label IS NULL
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


# def add_labels_to_catalog(catalog, subject_ids, subject_labels, id_col, label_col):
#     # better done with a database probably
#     labeller = dict(zip(subject_ids, subject_labels))
#     def add_label(row):
#         if row[id_col] in labeller.keys():
#             return labeller[row[id_col]]
#     catalog['label'] = catalog.apply(add_label, axis=1)
#     return catalog


def add_labelled_subjects_to_tfrecord(db, subject_ids, tfrecord_loc, size):
    cursor = db.cursor()

    # TODO move earlier, to get_top_acquisitions
    cursor.execute(
        '''
        SELECT id_str FROM catalog
        WHERE label IS NOT NULL
        LIMIT 3
        '''
    )
    if len(cursor.fetchall()) == 0:
        raise ValueError('Fatal: all subjects have labels in db. No more subjects to add!')

    rows = []
    for subject_id in subject_ids:
        assert isinstance(subject_id, str)
        logging.debug(subject_id)
        cursor.execute(
            '''
            SELECT id_str, label, fits_loc FROM catalog
            WHERE id_str = (:id_str)
            ''',
            (subject_id,)
        )
        # namedtuple would allow better type checking
        subject = cursor.fetchone()  # TODO ensure distinct
        if subject is None:
            raise ValueError('Fatal: top ids not found in catalog or label missing!')
        # if not isinstance(subject[0], str):
        #     raise ValueError('Fatal: {} subject id is not str!'.format(subject_id))
        # if not isinstance(subject[1], int):
        #     raise ValueError('Fatal: {} label is not int!'.format(subject_id))
        logging.debug(subject)
        if subject[1] is None:
            raise ValueError('Fatal: {} missing label in db!'.format(subject_id))
        assert subject[1] != b'\x00\x00\x00\x00\x00\x00\x00\x00'  # i.e. null label
        rows.append({
            'id_str': str(subject[0]),  # db cursor casts to int-like string to int...
            'label': int(subject[1]),
            'fits_loc': str(subject[2])
        })
    
    top_subject_df = pd.DataFrame(data=rows)
    print(top_subject_df.iloc[0])
    print(top_subject_df.iloc[1])
    catalog_to_tfrecord.write_image_df_to_tfrecord(
        top_subject_df, 
        tfrecord_loc, 
        size, 
        columns_to_save=['id_str', 'label'], 
        append=True, # must append, don't overwrite previous training data! 
        source='fits')


def setup(catalog, db_loc, id_col, size, shard_dir, shard_size):
    db = create_db(catalog, db_loc, id_col)
    columns_to_save = [id_col]
    write_catalog_to_tfrecord_shards(catalog, db, size, id_col, columns_to_save, shard_dir, shard_size=shard_size)


def run(catalog, db_loc, id_col, label_col, size, channels, predictor_dir, train_tfrecord_loc, train_callable):
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

        logging.debug('ids {}'.format(top_acquisition_ids))
        labels = mock_panoptes.get_labels(top_acquisition_ids)
        logging.debug('labels {}'.format(labels))

        add_labels_to_db(top_acquisition_ids, labels, db)

        logging.info('Saving top acquisition subjects to {}, labels: {}'.format(train_tfrecord_loc, labels))
        add_labelled_subjects_to_tfrecord(db, top_acquisition_ids, train_tfrecord_loc, size)

        iteration += 1


def add_labels_to_db(subject_ids, labels, db):
    cursor = db.cursor()

    for subject_n in range(len(subject_ids)):
        label = labels[subject_n]
        subject_id = subject_ids[subject_n]

        assert isinstance(label, np.int64) or isinstance(label, int)
        assert isinstance(subject_id, str)
        logging.debug('label {}, id {}'.format(label, subject_id))
        cursor.execute(
            '''
            UPDATE catalog
            SET label = (:label)
            WHERE id_str = (:subject_id)
            ''',
            (label, subject_id)
        )
        db.commit()

        # check labels really have been added
        cursor.execute(
            '''
            SELECT label FROM catalog
            WHERE id_str = (:id_str)
            LIMIT 1
            ''',
            (subject_id,)
        )
        retrieved_label = cursor.fetchone()[0]
        assert retrieved_label == label


def get_all_shard_locs(db):
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT DISTINCT tfrecord FROM shardindex
        ORDER BY tfrecord ASC
        '''
    )
    return [row[0] for row in cursor.fetchall()]  # list of shard locs
