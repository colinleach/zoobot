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
            label INT DEFAULT NULL,
            total_votes INT DEFAULT NULL,
            file_loc STRING)
        '''
    )
    db.commit()

    # no longer used
    cursor.execute(
        '''
        CREATE TABLE shardindex(
            id_str STRING PRIMARY KEY,
            tfrecord TEXT)
        '''
    )
    db.commit()

    # no longer used
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

    catalog_entries = list(df[['id_str', 'file_loc']].itertuples(index=False, name='CatalogEntry'))
    cursor = db.cursor()
    cursor.executemany(
        '''
        INSERT INTO catalog(id_str, label, total_votes, file_loc) VALUES(?,NULL,NULL,?)''',
        catalog_entries
    )
    db.commit()


def write_catalog_to_tfrecord_shards(df, db, img_size, columns_to_save, save_dir, shard_size=1000):
    """Write galaxy catalog of id_str and file_loc across many tfrecords, and record in db.
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
    assert all(column in df.columns.values for column in columns_to_save)

    df = df.copy().sample(frac=1).reset_index(drop=True)  # shuffle
    # split into shards
    shard_n = 0
    n_shards = (len(df) // shard_size) + 1
    df_shards = [df.iloc[n * shard_size:(n + 1) * shard_size] for n in range(n_shards)]

    for shard_n, df_shard in enumerate(df_shards):
        save_loc = os.path.join(save_dir, 's{}_shard_{}.tfrecord'.format(img_size, shard_n))
        catalog_to_tfrecord.write_image_df_to_tfrecord(
            df_shard, 
            save_loc,
            img_size,
            columns_to_save,
            reader=catalog_to_tfrecord.get_reader(df['file_loc']),
            append=False)
        if db is not None:  # explicitly not passing db will skip this step, for e.g. train/test
            add_tfrecord_to_db(save_loc, db, df_shard)  # record ids in db, not labels


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

# should make each shard a comparable size to the available memory, but can iterate over several if needed
def make_predictions_on_tfrecord(tfrecord_locs, model, db, n_samples, size, max_images=10000):
    """Record model predictions (samples) on a list of tfrecords, for each unlabelled galaxy.
    Load in batches to avoid maxing out GPU memory
    See make_predictions_on_tfrecord_batch for more details.
    Uses db to identify which subjects are unlabelled by matching on id_str feature.
    
    Args:
        tfrecord_locs (list): paths to tfrecords to load
        model (function): callable returning [batch, n_samples] predictions
        db (sqlite3): database of galaxies, with id_str and label columns. Not modified.
        n_samples (int): number of MC dropout samples to calculate (i.e. n forward passes)
        size (int): size of saved images
        max_images (int, optional): Defaults to 10000. Load no more than this many images per record
    
    Returns:
        list: unlabelled subjects of form [{id_str: .., matrix: ..}, ..]
        np.ndarray: predictions on those subjects, with shape [n_unlabelled_subjects, n_samples]
    """
    assert isinstance(tfrecord_locs, list)
    # TODO review this once images have been made smaller, may be able to have much bigger batches

    records_per_batch = 2  # best to be >= num. of CPU, for parallel reading. But at 256px, only fits 2 records.
    min_tfrecord = 0
    all_unlabelled_subjects = []  # list of generators for unlabelled subjects
    all_samples = []  # list of np arrays of samples

    while min_tfrecord < len(tfrecord_locs):
        unlabelled_subjects, samples = make_predictions_on_tfrecord_batch(
            tfrecord_locs[min_tfrecord:min_tfrecord + records_per_batch],
            model, db, n_samples, size, max_images=10000)
       
        all_unlabelled_subjects.extend(unlabelled_subjects)
        all_samples.append(samples)

        assert len(unlabelled_subjects) != 0

        min_tfrecord += records_per_batch

    all_samples_arr = np.concatenate(all_samples, axis=0)
    logging.debug('Finished predictions {} on subjects {}'.format(
        all_samples_arr.shape, len(all_unlabelled_subjects))
    )
    return all_unlabelled_subjects, all_samples_arr


def make_predictions_on_tfrecord_batch(tfrecords_batch_locs, model, db, n_samples, size, max_images=10000):
    """Record model predictions (samples) on a list of tfrecords, for each unlabelled galaxy.
    Uses db to identify which subjects are unlabelled by matching on id_str feature.
    Data in tfrecords_batch_locs must fit in memory; otherwise use `make_predictions_on_tfrecord`

    Args:
        tfrecords_batch_locs (list): paths to tfrecords to load
        model (function): callable returning [batch, samples] predictions
        db (sqlite3): database of galaxies, with id_str and label columns. Not modified.
        n_samples (int): number of MC dropout samples to calculate (i.e. n forward passes)
        size (int): size of saved images
        max_images (int, optional): Defaults to 10000. Load no more than this many images per record
    
    Returns:
        list: unlabelled subjects of form [{id_str: .., matrix: ..}, ..]
        np.ndarray: predictions on those subjects, with shape [n_unlabelled_subjects, n_samples]
    """
    logging.info('Predicting on {}'.format(tfrecords_batch_locs))
    batch_images, _, batch_id_str = input_utils.predict_input_func(
                tfrecords_batch_locs,
                n_galaxies=max_images,
                initial_size=size,
                mode='id_str'
            )
    with tf.Session() as sess:
        images, id_str_bytes = sess.run([batch_images, batch_id_str])
        if len(images) == max_images:
            logging.critical(
                'Warning! Shards are larger than memory! Loaded {} images'.format(max_images)
            )

    # tfrecord will have encoded to bytes, need to decode
    logging.debug('Constructing subjects from loaded data')
    subjects = (
        {'matrix': image, 'id_str': id_str.decode('utf-8')} 
        for image, id_str in zip(images, id_str_bytes)
    )  # generator expression to minimise memory
    # logging.info('Loaded {} subjects from {}'.format(len(subjects), (tfrecords_batch_locs)))
    del images  # free memory

    # exclude subjects with labels in db
    logging.info('Filtering for unlabelled subjects')
    if db_fully_labelled(db):
        logging.critical('All subjects are labelled - stop running active learning!')
        exit(0)
    unlabelled_subjects = [
        subject for subject in subjects
        if not subject_is_labelled(subject['id_str'], db)
    ]
    logging.info('Loaded {} unlabelled subjects from {} of size {}'.format(
        len(unlabelled_subjects),
        tfrecords_batch_locs,
         size)
        )
    assert unlabelled_subjects
    del subjects  # free memory

    # make predictions on only those subjects
    logging.debug('Extracting images from unlabelled subjects')
    # need to construct array from list: required to have known length
    unlabelled_subject_data = np.array([subject['matrix'] for subject in unlabelled_subjects])
    for subject in unlabelled_subjects:
        del subject['matrix']  # we don't need after this, so long as we skip recording state
    samples = make_predictions.get_samples_of_images(model, unlabelled_subject_data, n_samples)
    return unlabelled_subjects, samples


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
        SELECT id_str, label
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


def db_fully_labelled(db):
    # check that at least one subject has no label
    if not get_all_subjects(db, labelled=True):
        logging.warning('Shard appears to be completely labelled')
        return True
    else:
        return False  # technically, None is considered False, but this is type-consistent

# bad name, actually gets all table columns
def get_file_loc_df_from_db(db, subject_ids: list):
    """
    Look up the file_loc and label in db for the subjects with ids in `subject_ids`
    df will include the image_loc stored at `db.catalog.file_loc`
    df will include the label stored under `db.catalog.label`
    Useful for creating additional training tfrecords during active learning
    
    Args:
        db (sqlite3.Connection): database with `db.catalog` table to get subject image and label
        subject_ids (list): of subject ids matching `db.catalog.id_str` values
    
    Raises:
        IndexError: No matches found to an id in `subject_ids` within db.catalog.id_str
        ValueError: Subject with an id in `subject_ids` has null label (code error in this func.)
    """
    assert len(subject_ids) > 0
    cursor = db.cursor()

    rows = []
    for subject_id in subject_ids:
        # find subject data in db
        assert isinstance(subject_id, str)
        logging.debug(subject_id)
        cursor.execute(
            '''
            SELECT id_str, label, total_votes, file_loc FROM catalog
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
        # add subject data to df
        rows.append({
            'id_str': str(subject[0]),  # db cursor casts to int-like string to int...
            'label': int(subject[1]),
            'total_votes': int(subject[2]),
            'file_loc': str(subject[3])  # db must contain accurate path to image
        })
        assert os.path.isfile(str(subject[3]))  # check that image path is correct

    top_subject_df = pd.DataFrame(data=rows)
    assert len(top_subject_df) == len(subject_ids)
    return top_subject_df


def add_labelled_subjects_to_tfrecord(db, subject_ids, tfrecord_loc, size):
    # must only be called with subject_ids that are all labelled, else will raise error
    assert not os.path.isfile(tfrecord_loc)  # this will overwrite, tfrecord can't append
    df = get_file_loc_df_from_db(db, subject_ids)
    catalog_to_tfrecord.write_image_df_to_tfrecord(
        df,
        tfrecord_loc,
        size,
        columns_to_save=['id_str', 'label', 'total_votes'],
        reader=catalog_to_tfrecord.get_reader(df['file_loc']))


def get_latest_checkpoint_dir(base_dir):
    saved_models = [x for x in os.listdir(base_dir) if x.startswith('15')]  # subfolders with timestamps
    saved_models.sort(reverse=True)  # sort by name i.e. timestamp, early timestamp first
    return os.path.join(base_dir, saved_models[0])  # the subfolder with the most recent time


def add_labels_to_db(subject_ids, labels, total_votes, db):
    # be careful: don't update any labels that might already have been written to tfrecord!
    logging.info('Adding new labels for {} subjects to db'.format(len(subject_ids)))
    logging.debug('Example subject ids: {}'.format(subject_ids[:3]))
    logging.debug('Example labels: {}'.format(labels[:3]))
    logging.debug('Example total_votes: {}'.format(total_votes[:3]))
    assert len(subject_ids) == len(labels) == len(total_votes)

    cursor = db.cursor()
    for subject_n in range(len(subject_ids)):
        label = labels[subject_n]
        total_votes_val = total_votes[subject_n]
        subject_id = subject_ids[subject_n]

        # np.int64 is wrongly written as byte string e.g. b'\x00...',  b'\x01...'
        label = int(label)
        total_votes_val = int(total_votes_val)
        assert isinstance(label, int)
        assert isinstance(label, int)
        assert isinstance(subject_id, str)

        # check not already labelled, else raise a manual error (see below)
        cursor.execute(
            '''
            SELECT id_str, label FROM catalog
            WHERE id_str = (:subject_id) AND label is NOT NULL
            ''',
            (subject_id,)
        )
        if cursor.fetchone() is not None:
            raise ValueError(
                'Trying to set label {} for already-labelled subject {}'.format(label, subject_id)
            )

        # set the label (this won't raise an automatic error if already exists!)
        cursor.execute(
            '''
            UPDATE catalog
            SET label = (:label), total_votes = (:total_votes)
            WHERE id_str = (:subject_id)
            ''',
            {
                'label': label,
                'subject_id': subject_id,
                'total_votes': total_votes_val
            }
        )
        db.commit()

        if subject_n == 0:  # careful check on first write only, for speed
            # check labels really have been added, and not as byte string
            cursor.execute(
                '''
                SELECT label, total_votes FROM catalog
                WHERE id_str = (:subject_id)
                LIMIT 1
                ''',
                (subject_id,)
            )
            row = cursor.fetchone()
            assert row is not None
            retrieved_label = row[0]
            assert retrieved_label == label
            retrieved_total_votes = row[1]
            assert retrieved_total_votes == total_votes_val


def get_all_subjects(db, labelled=None):
    cursor = db.cursor()
    if labelled:
        cursor.execute(
            '''
            SELECT id_str, label FROM catalog
            WHERE label IS NOT NULL
            '''
        )
    if labelled == False:  # explicitly not None
        cursor.execute(
            '''
            SELECT id_str FROM catalog
            WHERE label IS NULL
            '''
        )
    else:
        cursor.execute(
            '''
            SELECT id_str FROM catalog
            '''
        )
    result = cursor.fetchall()
    # assert False
    if result:
        return [x[0] for x in result]  # id_strs
    else:
        return []


def get_all_shard_locs(db):
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT DISTINCT tfrecord FROM shardindex
        
        ORDER BY tfrecord ASC
        '''
    )
    return [row[0] for row in cursor.fetchall()]  # list of shard locs


def filter_for_new_only(db, all_subject_ids, all_labels, all_total_votes):
    # TODO needs test
    # TODO wrap oracle subject as namedtuple?

    all_subjects = get_all_subjects(db)  # strictly, all sharded subjects - ignore train/eval catalog entries
    logging.info('all_subject_ids, {}'.format(all_subject_ids[:3]))
    logging.info('all_subjects, {}, of {}'.format(all_subjects[:3], len(all_subjects)))
    logging.info('matched: {}'.format(set(all_subject_ids).intersection(all_subjects)))
    found_in_db = [x in all_subjects for x in all_subject_ids]
    assert found_in_db  # should always be some galaxies not in train or eval, even if labelled
    logging.info('Oracle-labelled subjects in db: {}'.format(sum(found_in_db)))

    labelled_subjects = get_all_subjects(db, labelled=True)
    not_yet_labelled = [x not in labelled_subjects for x in all_subject_ids]
    logging.info('Not yet labelled (or maybe not matched): {}'.format(sum(not_yet_labelled)))

    # lists can't be boolean indexed, so convert to old-fashioned numeric index...
    indices = np.array([n for n in range(len(all_subject_ids))])
    safe_to_label = np.array(found_in_db) & np.array(not_yet_labelled)
    indices_safe_to_label = indices[safe_to_label]
    if sum(safe_to_label) == len(all_subject_ids):
        logging.warning('All oracle labels identified as new - does this make sense?')
    logging.info(
        'Galaxies to newly label: {} of {}'.format(len(indices_safe_to_label), len(all_subject_ids))
    )
    # ...then use list comprehension to select with numeric index
    safe_subject_ids = [all_subject_ids[i] for i in indices_safe_to_label]
    safe_labels = [all_labels[i] for i in indices_safe_to_label]
    safe_total_votes = [all_total_votes[i] for i in indices_safe_to_label]
    return safe_subject_ids, safe_labels, safe_total_votes
