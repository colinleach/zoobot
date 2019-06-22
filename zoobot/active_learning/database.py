"""
Everything that happens at subject/database level: taking subject data, imperative db commands
If swapping the database for another database tool, or different storage method, would change anything, 
it belongs in db_access not in here.

This contains code that the rest of the package uses to do active learning things requiring the database
All package code using the database can only use this file

This file turns those package requests into database actions
However, it is agnostic to how those actions are implemented - the db could be swapped
That's what db_access.py is for
"""
import os
from typing import List
import logging
from collections import Counter

import numpy as np
import tensorflow as tf
import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord, create_tfrecord, read_tfrecord
from zoobot.estimators import input_utils, estimator_params, run_estimator, make_predictions
from zoobot.active_learning import db_access


class Subject:

    def __init__(self, catalog_entry: db_access.CatalogEntry):
        assert isinstance(catalog_entry, db_access.CatalogEntry)
        self.id_str = catalog_entry.id_str
        self.file_loc = catalog_entry.file_loc
        self.labels = catalog_entry.labels

    def unpacked(self):
        data = {
            'id_str': self.id_str,
            'file_loc': self.file_loc,
        }
        data.update(**self.labels)
        return data


def get_all_subjects_df(db, labelled=None):
    """Reverse the database construction: get back the subjects and labels, column-by-column
    
    Args:
        db ([type]): [description]
        labelled ([type], optional): [description]. Defaults to None.
    
    Returns:
        [type]: [description]
    """
    subjects = get_all_subjects(db, labelled)
    return subjects_to_dataframe(subjects)


def get_all_subjects(db, labelled):
    entries = db_access.get_all_entries(db, labelled)
    return [Subject(x) for x in entries]

def get_specific_subjects_df(db, subject_ids):
    subjects = get_specific_subjects(db, subject_ids)
    return subjects_to_dataframe(subjects)

def get_specific_subjects(db, subject_ids):
    entries = [db_access.get_entry(db, subject_id) for subject_id in subject_ids]
    return [Subject(x) for x in entries]

def subjects_to_dataframe(subjects):
    return pd.DataFrame([x.unpacked() for x in subjects])


def db_fully_labelled(db):
    # check that at least one subject has no labels
    if not len(db_access.get_all_entries(db, labelled=False)):
        logging.warning('Database appears to be completely labelled')
        return True
    else:
        return False  # technically, None is considered False, but this is type-consistent

def filter_for_new_only(db, subject_ids_to_filter: List, all_labels: List):
    # TODO wrap oracle subject as namedtuple?
    all_subject_ids_in_db = [x.id_str for x in db_access.get_all_entries(db)]  # strictly, all sharded subjects - ignore train/eval catalog entries
    logging.info('subject_ids_to_filter, {}'.format(subject_ids_to_filter[:3]))
    logging.info('all_subject_ids_in_db, {}, of {}'.format(all_subject_ids_in_db[:3], len(all_subject_ids_in_db)))
    logging.info('matched: {}'.format(set(subject_ids_to_filter).intersection(all_subject_ids_in_db)))
    found_in_db = [x in all_subject_ids_in_db for x in subject_ids_to_filter]
    assert found_in_db  # should always be some galaxies not in train or eval, even if labelled
    logging.info('Oracle-labelled subjects in db: {}'.format(sum(found_in_db)))

    labelled_id_strs_in_db = [x.id_str for x in db_access.get_all_entries(db, labelled=True)]
    not_yet_labelled = [x not in labelled_id_strs_in_db for x in subject_ids_to_filter]
    logging.info('Not yet labelled (or maybe not matched): {}'.format(sum(not_yet_labelled)))

    # lists can't be boolean indexed, so convert to old-fashioned numeric index...
    indices = np.array([n for n in range(len(subject_ids_to_filter))])
    safe_to_label = np.array(found_in_db) & np.array(not_yet_labelled)
    indices_safe_to_label = indices[safe_to_label]
    if sum(safe_to_label) == len(subject_ids_to_filter):
        logging.warning('All oracle labels identified as new - does this make sense?')
    logging.info(
        'Galaxies to newly label: {} of {}'.format(len(indices_safe_to_label), len(subject_ids_to_filter))
    )
    # ...then use list comprehension to select with numeric index
    safe_subject_ids = [subject_ids_to_filter[i] for i in indices_safe_to_label]
    safe_labels = [all_labels[i] for i in indices_safe_to_label]
    return safe_subject_ids, safe_labels



def verify_db_matches_catalog(labelled_catalog, db):
     # db should contain the catalog in 'catalog' table
    subjects = get_all_subjects(db, labelled=None)
    for subject in subjects:
        recovered_id = subject['id_str']
        recovered_loc = subject['file_loc']
        expected_loc = labelled_catalog[labelled_catalog['id_str'] == recovered_id].squeeze()['file_loc']
        assert recovered_loc == expected_loc


def verify_db_matches_shards(db, size, channels):
    # db should contain file locs in 'shardindex' table
    # tfrecords should have been written with the right files
    shardindex = db_access.load_shards(db)
    tfrecord_locs = shardindex['tfrecord_loc'].unique()
    for tfrecord_loc in tfrecord_locs:
        expected_shard_ids = set(shardindex[shardindex['tfrecord_loc'] == tfrecord_loc]['id_str'].unique())
        examples = read_tfrecord.load_examples_from_tfrecord(
            [tfrecord_loc], 
            read_tfrecord.matrix_id_feature_spec(size, channels)
        )
        actual_shard_ids = set([example['id_str'].decode() for example in examples])
        assert expected_shard_ids == actual_shard_ids


def verify_catalog_matches_shards(unlabelled_catalog, db, size, channels):
    shardindex = db_access.load_shards(db)
    tfrecord_locs = shardindex['tfrecord_loc'].unique()
    # check that every catalog id is in exactly one shard
    assert not any(unlabelled_catalog['id_str'].duplicated())  # catalog must be unique to start with
    catalog_ids = Counter(unlabelled_catalog['id_str'])  # all 1's
    shard_ids = Counter()

    for tfrecord_loc in tfrecord_locs:
        examples = read_tfrecord.load_examples_from_tfrecord(
            [tfrecord_loc],
            read_tfrecord.matrix_id_feature_spec(size, channels)
        )
        ids_in_shard = [x['id_str'].decode() for x in examples]
        assert len(ids_in_shard) == len(set(ids_in_shard))  # must be unique within shard
        shard_ids = Counter(ids_in_shard) + shard_ids

    assert catalog_ids == shard_ids



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
            db_access.add_tfrecord_to_db(save_loc, db, df_shard)  # record ids in db, not labels


def add_labelled_subjects_to_tfrecord(db, subject_ids, tfrecord_loc, size, columns_to_save):
    # must only be called with subject_ids that are all labelled, else will raise error
    assert not os.path.isfile(tfrecord_loc)  # this will overwrite, tfrecord can't append
    df = get_specific_subjects_df(db, subject_ids)
    catalog_to_tfrecord.write_image_df_to_tfrecord(
        df,
        tfrecord_loc,
        size,
        columns_to_save=columns_to_save,
        reader=catalog_to_tfrecord.get_reader(df['file_loc']))


# should make each shard a comparable size to the available memory, but can iterate over several if needed
def make_predictions_on_tfrecord(tfrecord_locs, model, db, n_samples, size, max_images=10000):
    """Record model predictions (samples) on a list of tfrecords, for each unlabelled galaxy.
    Load in batches to avoid maxing out GPU memory
    See make_predictions_on_tfrecord_batch for more details.
    Uses db to identify which subjects are unlabelled by matching on id_str feature.
    
    Args:
        tfrecord_locs (list): paths to tfrecords to load
        model (function): callable returning [batch, n_samples] predictions
        db (sqlite3): database of galaxies, with id_str and labels columns. Not modified.
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
        db (sqlite3): database of galaxies, with id_str and labels columns. Not modified.
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
        if not db_access.subject_is_labelled(subject['id_str'], db)
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
    samples = db_access.make_predictions.get_samples_of_images(model, unlabelled_subject_data, n_samples)
    return unlabelled_subjects, samples

# TODO?
get_all_shard_locs = db_access.get_all_shard_locs
