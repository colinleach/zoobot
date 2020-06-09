"""Implementation details of how the database works to store subjects.
Should ONLY be accessed via database.py"""
import logging
import os
import shutil
import functools
import itertools
import sqlite3
import time
import json
from collections import namedtuple
from typing import List

import tensorflow as tf
import pandas as pd
import numpy as np

from zoobot import shared_utilities
from zoobot.estimators import estimator_params, make_predictions


CatalogEntry = namedtuple('CatalogEntry', ['id_str', 'file_loc', 'labels'])

def create_db(catalog, db_loc, timeout=15.0):
    """Instantiate sqlite database at db_loc with the following tables:
    1. `catalog`: analogy of catalog dataframe, for fits locations and (sometimes) labels
    2. `shards`: which shard contains each (serialized) galaxy
    3. `acquisitions`: the latest acquistion function value for each galaxy
    
    Args:
        catalog (pd.DataFrame): with id_col (id string) and fits_loc (image on disk) columns
        db_loc (str): file location to save database
    
    Returns:
        sqlite3.Connection: connection to database as described above. Intended for active learning
    """
    db = sqlite3.connect(db_loc, timeout=timeout, isolation_level='IMMEDIATE')

    cursor = db.cursor()

    # columns [i.e. catalog columns, usually including labels] are stored as json-serialized dicts 
    cursor.execute(
        '''
        CREATE TABLE catalog(
            id_str STRING PRIMARY KEY,
            labels STRING DEFAULT NULL,
            file_loc STRING)
        '''
    )
    db.commit()

    # no longer used - but leave as it may prove useful
    cursor.execute(
        '''
        CREATE TABLE shards(
            id_str STRING PRIMARY KEY,
            tfrecord_loc TEXT)
        '''
    )
    db.commit()


    add_catalog_to_db(catalog, db)  # does NOT add labels, labels are unknown at this point

    return db


def add_catalog_to_db(df, db):
    """Add id and fits image location info from galaxy catalog to db `catalog` table
    
    Args:
        df (pd.DataFrame): Galaxy catalog with 'id_str' and 'fits_loc' fields. No other cols saved.
        db (sqlite3.Connection): database with `catalog` table to record df id_col and fits_loc
    """
    catalog_entries = list(df[['id_str', 'file_loc']].itertuples(index=False, name='CatalogEntry'))
    cursor = db.cursor()
    cursor.executemany(
        '''
        INSERT INTO catalog(id_str, file_loc, labels) VALUES(?,?, NULL)''',
        catalog_entries
    )
    db.commit()


def add_tfrecord_to_db(tfrecord_loc, db, df):
    """Update db to record catalog entries as inside a tfrecord.
    Note: does not actually load/verify the tfrecord, simply takes info from catalog
    TODO scan through the record rather than just reading df?
    TODO eventually, consider the catalog being SQL as source-of-truth and csv output

    Args:
        tfrecord_loc (str): disk path of tfrecord
        db (sqlite3.Connection): database with `catalog` table to record df id_col and fits_loc
        df (pd.DataFrame): Galaxy catalog with 'id_str' and 'fits_loc' fields
    """
    shard_index_entries = list(zip(
        df['id_str'].values, 
        [tfrecord_loc for n in range(len(df))]
    ))
    cursor = db.cursor()
    cursor.executemany(
        '''
        INSERT INTO shards(id_str, tfrecord_loc) VALUES(?,?)''',
        shard_index_entries
    )
    db.commit()

# def save_acquisition_to_db(subject_id, acquisition, db): 
#     """Save the acquisition value for the subject to the database
#     Warning: will overwrite previous acquisitions for that subject
    
#     Args:
#         subject_id (str): id string of subject, expected to match db.acquisitions.id_str values
#         acquisition (float): latest acquisition value for subject with `subject_id` id string
#         db (sqlite3.Connection): database with `acquisitions` table to record acquisition
#     """
#     assert isinstance(acquisition, float)  # can't be np.float32, else written as bytes
#     assert isinstance(subject_id, str)
#     cursor = db.cursor()  
#     cursor.execute('''  
#     INSERT OR REPLACE INTO acquisitions(id_str, acquisition_value)  
#                   VALUES(:id_str, :acquisition_value)''',
#                   {
#                       'id_str': subject_id, 
#                       'acquisition_value': acquisition})
#     db.commit()


def add_labels_to_db(subject_ids: List, all_labels: List, db):
    """[summary]
    
    Args:
        subject_ids (List): [description]
        all_labels (List): each set of labels (element) will be dumped to json
        db ([type]): [description]
    
    Raises:
        a: [description]
        ValueError: [description]
        an: [description]
    """
    # be careful: don't update any labels that might already have been written to tfrecord!
    logging.info('Adding new labels for {} subjects to db'.format(len(subject_ids)))
    logging.debug('Example subject ids: {}'.format(subject_ids[:3]))
    logging.debug('Example labels: {}'.format(all_labels[:3]))
    assert len(subject_ids) == len(all_labels)

    cursor = db.cursor()
    for subject_n in range(len(subject_ids)):
        labels = all_labels[subject_n]
        subject_id = subject_ids[subject_n]

        assert isinstance(subject_id, str)
        labels_str = json.dumps(labels)
        
        # check not already labelled, else raise a manual error (see below)
        cursor.execute(
            '''
            SELECT id_str, labels FROM catalog
            WHERE id_str = (:subject_id) AND labels is NOT NULL
            ''',
            (subject_id,)
        )
        if cursor.fetchone() is not None:
            raise ValueError(
                'Trying to set labels {} for already-labelled subject {}'.format(labels, subject_id)
            )
        logging.info('{} {} {}'.format(subject_n, subject_id, labels))  # temporary
        # the logging seemed to help, so maybe it's just falling over itself a little bit?
        time.sleep(0.002) # 2ms, typical write time 30ms
        # set the label (this won't raise an automatic error if already exists!)
        cursor.execute(
            '''
            UPDATE catalog
            SET labels = (:labels_str)
            WHERE id_str = (:subject_id)
            ''',
            {
                'labels_str': labels_str,
                'subject_id': subject_id
            }
        )
        db.commit()

        if subject_n == 0:  # careful check on first write only, for speed
            # check labels really have been added, and not as byte string
            cursor.execute(
                '''
                SELECT labels FROM catalog
                WHERE id_str = (:subject_id)
                LIMIT 1
                ''',
                (subject_id,)
            )
            row = cursor.fetchone()
            assert row is not None
            retrieved_labels = row[0]
            assert retrieved_labels == labels_str

    logging.info('{} labels added to db'.format(len(len(subject_ids))))


def get_all_entries(db, labelled=None):
    cursor = db.cursor()
    if labelled:
        cursor.execute(
            '''
            SELECT id_str, file_loc, labels FROM catalog
            WHERE labels IS NOT NULL
            '''
        )
    elif labelled == False:  # explicitly not None
        cursor.execute(
            '''
            SELECT id_str, file_loc, labels FROM catalog
            WHERE labels IS NULL
            '''
        )
    else:
        cursor.execute(
            '''
            SELECT id_str, file_loc, labels FROM catalog
            '''
        )
    results = cursor.fetchall()
    return [CatalogEntry(*result) for result in results]


def get_entry(db, subject_id):
    # find subject data in db
    assert isinstance(subject_id, str)
    logging.debug(subject_id)
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT id_str, labels, file_loc FROM catalog
        WHERE id_str = (:id_str)
        ''',
        (subject_id,)
    )
    subject = cursor.fetchone()  # TODO ensure distinct
    if subject is None:
        raise IndexError('Fatal: top ids not found in catalog or labels missing!')
    if subject[1] is None:
        raise ValueError('Fatal: {} missing labels in db!'.format(subject_id))
    if not os.path.isfile(str(subject[2])):  # check that image path is correct
        raise ValueError('Fatal: missing subject allegedly at {}'.format(str(subject[2])))

    return CatalogEntry(
        id_str=str(subject[0]),  # db cursor casts to int-like string to int...
        labels=json.loads(subject[1]),
        file_loc=str(subject[2])  # db must contain accurate path to image
    )


def get_all_shard_locs(db):
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT DISTINCT tfrecord_loc FROM shards
        
        ORDER BY tfrecord_loc ASC
        '''
    )
    return [row[0] for row in cursor.fetchall()]  # list of shard locs


def load_shards(db):
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT id_str, tfrecord_loc FROM shards
        '''
    )
    shard_entries = cursor.fetchall()
    shard_data = []
    for entry in shard_entries:
        shard_data.append({
            'id_str': str(entry[0]),  # TODO shardindex id is str, loaded from byte string
            'tfrecord_loc': entry[1]
        })
    return pd.DataFrame(data=shard_data)


def subject_is_labelled(id_str: str, db):
    """Get the subject with subject_id

    Args:
        id_str (str): subject_id to search for
        db (sqlite3.Connection): database with `catalog` table to read labels

    Raises:
        ValueError: all subjects in `db.catalog` have labels

    Returns:
        bool: subject with id_str is labelled
    """
    # find the subject(s) with id_str in db
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT id_str, labels
        FROM catalog
        WHERE id_str IS (:id_str)
        ''',
        (id_str,)
    )
    matching_subjects = cursor.fetchall()  # sqlite3 may autoconvert id_str to int at this step
    if len(matching_subjects) < 1:
        raise ValueError('Subject not found: {}'.format(id_str))
    if len(matching_subjects) > 1:
        raise ValueError('Duplicate subject in db: {}'.format(id_str))
    # logging.debug(matching_subjects)
    # logging.debug(matching_subjects[0][1])
    return matching_subjects[0][1] is not None

