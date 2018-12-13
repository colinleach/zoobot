import logging
import os
import shutil
import functools
import itertools
import sqlite3
import time
import json
from collections import namedtuple

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


def create_db(catalog, db_loc):
    """Instantiate sqlite database at db_loc with the following tables:
    1. `catalog`: analogy of catalog dataframe, for fits locations and (sometimes) labels
    2. `shardindex`: which shard contains each (serialized) galaxy
    3. `acquisitions`: the latest acquistion function value for each galaxy
    
    Args:
        catalog (pd.DataFrame): with id_col (id string) and fits_loc (image on disk) columns
        db_loc (str): file location to save database
    
    Returns:
        sqlite3.Connection: connection to database as described above. Intended for active learning
    """
    db = sqlite3.connect(db_loc)

    cursor = db.cursor()

    cursor.execute(
        '''
        CREATE TABLE catalog(
            id_str STRING PRIMARY KEY,
            label FLOAT DEFAULT NULL,
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

    add_catalog_to_db(catalog, db)  # does NOT add labels, labels are unknown at this point

    return db


def add_catalog_to_db(df, db):
    """Add id and fits image location info from galaxy catalog to db `catalog` table
    
    Args:
        df (pd.DataFrame): Galaxy catalog with 'id_str' and 'fits_loc' fields
        db (sqlite3.Connection): database with `catalog` table to record df id_col and fits_loc
    """

    catalog_entries = list(df[['id_str', 'fits_loc']].itertuples(index=False, name='CatalogEntry'))
    cursor = db.cursor()
    cursor.executemany(
        '''
        INSERT INTO catalog(id_str, label, fits_loc) VALUES(?,NULL,?)''',
        catalog_entries
    )
    db.commit()


def write_catalog_to_tfrecord_shards(df, db, img_size, columns_to_save, save_dir, shard_size=1000):
    """Write galaxy catalog across many tfrecords.
    Useful to quickly load images for repeated predictions.

    Args:
        df (pd.DataFrame): Galaxy catalog with 'id_str' and 'fits_loc' fields
        db (sqlite3.Connection): database with `catalog` table to record df id_col and fits_loc
        img_size (int): height/width dimension of image matrix to rescale and save to tfrecords
        columns_to_save (list): Catalog data to save with each subject. Names will match tfrecord.
        save_dir (str): disk directory path into which to save tfrecords
        shard_size (int, optional): Defaults to 1000. Max subjects per shard. Final shard has less.
    """
    assert not df.empty
    assert 'id_str' in columns_to_save

    df = df.copy().sample(frac=1).reset_index(drop=True)  # shuffle
    # split into shards
    shard_n = 0
    n_shards = (len(df) // shard_size) + 1
    df_shards = [df.iloc[n * shard_size:(n + 1) * shard_size] for n in range(n_shards)]

    for shard_n, df_shard in enumerate(df_shards):
        save_loc = os.path.join(save_dir, 's{}_shard_{}.tfrecord'.format(img_size, shard_n))
        catalog_to_tfrecord.write_image_df_to_tfrecord(df_shard, save_loc, img_size, columns_to_save, append=False, source='fits')
        add_tfrecord_to_db(save_loc, db, df_shard)


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
        INSERT INTO shardindex(id_str, tfrecord) VALUES(?,?)''',
        shard_index_entries
    )
    db.commit()


def record_acquisitions_on_tfrecord(db, tfrecord_loc, size, channels, acquisition_func):
    """For every subject in tfrecord, get the acq. func value and save to db
    Records acq. func. value on all examples, even labelled ones.
    Acquisition func 
    Note: Could cross-ref with db to skip predicting on labelled examples, but still need to load
    
    Args:
        db (sqlite3.Connection): database with `acquisitions` table to record aqf. func. value
        tfrecord_loc (str): disk path to tfrecord. Loaded tfrecord must fit in memory.
        size (int): height/width dimension of each image matrix to load from tfrecord
        channels (int): channels of each image matrix to load from tfrecord
        acquisition_func (callable): expecting list of image matrices, returning list of scalars
    """
    subjects = read_tfrecord.load_examples_from_tfrecord(
        [tfrecord_loc],
        read_tfrecord.matrix_id_feature_spec(size, channels)
    )
    logging.debug('Loaded {} subjects from {} of size {}'.format(len(subjects), tfrecord_loc, size))
    # acq func expects a list of matrices
    subjects_data = [x['matrix'].reshape(size, size, channels) for x in subjects]
    acquisitions = acquisition_func(subjects_data)  # returns list of acquisition values

    for subject, acquisition in zip(subjects, acquisitions):
        subject_id = subject['id_str'].decode('utf-8')  # tfrecord will have encoded to bytes
        save_acquisition_to_db(subject_id, acquisition, db)


def save_acquisition_to_db(subject_id, acquisition, db): 
    """Save the acquisition value for the subject to the database
    Warning: will overwrite previous acquisitions for that subject
    
    Args:
        subject_id (str): id string of subject, expected to match db.acquisitions.id_str values
        acquisition (float): latest acquisition value for subject with `subject_id` id string
        db (sqlite3.Connection): database with `acquisitions` table to record acquisition
    """
    assert isinstance(acquisition, float)  # can't be np.float32, else written as bytes
    assert isinstance(subject_id, str)
    cursor = db.cursor()  
    cursor.execute('''  
    INSERT OR REPLACE INTO acquisitions(id_str, acquisition_value)  
                  VALUES(:id_str, :acquisition_value)''',
                  {
                      'id_str': subject_id, 
                      'acquisition_value': acquisition})
    db.commit()


def get_top_acquisitions(db, n_subjects=1000, shard_loc=None):
    """Get the subject ids of up to `n_subjects`:
        1. Without labels (required for later training)
        1. With the highest acquisition values, up to the first `n_subjects`

    Args:
        db (sqlite3.Connection): database with `acquisitions` table to read acquisition
        n_subjects (int, optional): Defaults to 1000. Max subject ids to return.
        shard_loc (str, optional): Defaults to None. Get top subjects from only this shard.

    Raises:
        ValueError: all subjects in `db.catalog` have labels
        IndexError: if no top subjects are found, optionally looking only in shard_loc

    Returns:
        list: ordered list (descending) of subject id strings with highest acquisition values
    """
    cursor = db.cursor()
    # check that at least 1 subject has no label
    cursor.execute(
        '''
        SELECT id_str FROM catalog
        WHERE label IS NULL
        '''
    )
    unlabelled_subject = cursor.fetchone()
    if unlabelled_subject is None:
        raise ValueError('Fatal: all subjects have labels in db. No more subjects to add!')

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
        # first, verify that shard_loc is in at least one row of shardindex table
        cursor.execute(
            '''
            SELECT tfrecord from shardindex
            WHERE tfrecord = (:shard_loc)
            ''',
            (shard_loc,)
        )
        rows_in_shard = cursor.fetchone()
        assert rows_in_shard is not None
        # since the shard checks out, get top unlabelled subjects from that shard only
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

    top_subjects = cursor.fetchall()  # sqlite3 may autoconvert id_str to int at this step
    top_acquisitions = [str(x[0]) for x in top_subjects]  # to avoid, ensure id_str really is str
    if len(top_acquisitions) == 0:
        raise IndexError(
            'Fatal: No top subjects. Perhaps no unlabelled subjects in shard {} ?'.format(shard_loc)
            )
    return top_acquisitions


def add_labelled_subjects_to_tfrecord(db, subject_ids, tfrecord_loc, size):
    """Write the subjects with ids in `subject_ids` to a tfrecord at `tfrecord_loc`
    tfrecord will save the image stored at `db.catalog.fits_loc` at size `size`
    tfrecord will include the label stored under `db.catalog.label`
    Useful for creating additional training tfrecords during active learning
    
    Args:
        db (sqlite3.Connection): database with `db.catalog` table to get subject image and label
        subject_ids (list): of subject ids matching `db.catalog.id_str` values
        tfrecord_loc (str): path into which to save new tfrecord
        size (int): height/width dimension of image matrix to rescale and save to tfrecords
    
    Raises:
        IndexError: No matches found to an id in `subject_ids` within db.catalog.id_str
        ValueError: Subject with an id in `subject_ids` has null label (code error in this func.)
    """

    logging.info('Adding {} subjects  (e.g. {}) to new tfrecord {}'.format(len(subject_ids), subject_ids[:5], tfrecord_loc))
    assert not os.path.isfile(tfrecord_loc)  # this will overwrite, tfrecord can't append
    cursor = db.cursor()

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
            raise IndexError('Fatal: top ids not found in catalog or label missing!')

        if subject[1] is None:
            raise ValueError('Fatal: {} missing label in db!'.format(subject_id))
        assert subject[1] != b'\x00\x00\x00\x00\x00\x00\x00\x00'  # i.e. np.int64 write error
        rows.append({
            'id_str': str(subject[0]),  # db cursor casts to int-like string to int...
            'label': float(subject[1]),
            'fits_loc': str(subject[2])
        })

    top_subject_df = pd.DataFrame(data=rows)
    assert len(top_subject_df) == len(subject_ids)
    catalog_to_tfrecord.write_image_df_to_tfrecord(
        top_subject_df,
        tfrecord_loc,
        size,
        columns_to_save=['id_str', 'label'],
        source='fits')


def get_latest_checkpoint_dir(base_dir):
        saved_models = os.listdir(base_dir)  # subfolders
        saved_models.sort(reverse=True)  # sort by name i.e. timestamp, early timestamp first
        return os.path.join(base_dir, saved_models[-1])  # the subfolder with the most recent time


def add_labels_to_db(subject_ids, labels, db):
    cursor = db.cursor()

    # cursor.execute(
    #     '''
    #     SELECT * FROM catalog
    #     LIMIT 50
    #     '''
    # )

    for subject_n in range(len(subject_ids)):
        label = labels[subject_n]
        subject_id = subject_ids[subject_n]

        # np.int64 is wrongly written as byte string e.g. b'\x00...',  b'\x01...'
        if isinstance(label, np.float32):
            label = float(label)
        assert isinstance(label, float)
        assert isinstance(subject_id, str)

        cursor.execute(
            '''
            UPDATE catalog
            SET label = (:label)
            WHERE id_str = (:subject_id)
            ''',
            {
                'label': label,
                'subject_id': subject_id
            }
        )
        db.commit()

        # check labels really have been added, and not as byte string
        cursor.execute(
            '''
            SELECT label FROM catalog
            WHERE id_str = (:subject_id)
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
