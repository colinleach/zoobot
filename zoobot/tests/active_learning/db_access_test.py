import pytest

import logging
import os
import random
import sqlite3
import time
import json

import numpy as np
import tensorflow as tf
import pandas as pd
from astropy.io import fits

from zoobot.tfrecord import create_tfrecord, read_tfrecord
from zoobot.estimators.estimator_params import default_four_layer_architecture, default_params
from zoobot.active_learning import db_access
from zoobot.estimators import make_predictions
from zoobot.tests.active_learning import conftest


def test_add_catalog_to_db(empty_shard_db):
    minimal_df = pd.DataFrame(data=[
        {
            'id_str': 'some_hash',
            'file_loc': 'some_loc',
            'column_a': 'value_aa',
            'column_b': 'value_ab'
        },
        {
            'id_str': 'some_other_hash',
            'file_loc': 'some_other_loc',
            'column_a': 'value_ba',
            'column_b': 'value_bb'
        }
    ])
    db_access.add_catalog_to_db(minimal_df, empty_shard_db)
    cursor = empty_shard_db.cursor()
    cursor.execute(
        '''
        SELECT id_str, file_loc, labels FROM catalog
        '''
    )
    results = cursor.fetchall()
    for n, x in enumerate(results):
        assert x[0] == minimal_df.iloc[n]['id_str']
        assert x[1] == minimal_df.iloc[n]['file_loc']
        # assert x[2] == json.dumps(minimal_df.iloc[n][['column_a', 'column_b']])


def test_add_tfrecord_to_db(tfrecord_matrix_ints_loc, empty_shard_db, unlabelled_catalog):  # bad loc
    db_access.add_tfrecord_to_db(tfrecord_matrix_ints_loc, empty_shard_db, unlabelled_catalog)
    cursor = empty_shard_db.cursor()
    cursor.execute(
        '''
        SELECT id_str, tfrecord_loc FROM shards
        '''
    )
    saved_subjects = cursor.fetchall()
    for n, subject in enumerate(saved_subjects):
        assert str(subject[0]) == unlabelled_catalog.iloc[n]['id_str']  # strange string casting when read back
        assert subject[1] == tfrecord_matrix_ints_loc


# def test_save_acquisition_to_db(unknown_subject, acquisition, empty_shard_db):
#     active_learning.save_acquisition_to_db(unknown_subject['id_str'], acquisition, empty_shard_db)
#     cursor = empty_shard_db.cursor()
#     cursor.execute(
#         '''
#         SELECT id_str, acquisition_value FROM acquisitions
#         '''
#     )
#     saved_subject = cursor.fetchone()
#     assert saved_subject[0] == unknown_subject['id_str']
#     assert np.isclose(saved_subject[1], acquisition)


def test_subject_is_labelled(filled_shard_db_with_partial_labels):
    id_strs = ['some_hash', 'some_other_hash', 'yet_another_hash']
    labelled_ids = [db_access.subject_is_labelled(id_str, filled_shard_db_with_partial_labels) for id_str in id_strs]
    assert labelled_ids == [True, False, False]


def test_subject_is_labelled_missing_subject(filled_shard_db_with_partial_labels):
    with pytest.raises(ValueError):
        db_access.subject_is_labelled('missing_subject', filled_shard_db_with_partial_labels)


def test_add_labels_to_db(filled_shard_db):
    subjects = [
        {
            'id_str': 'some_hash',
            'labels': {'column': 1}
        },
        {
            'id_str': 'yet_another_hash',
            'labels': {'column': 1}
        }
    ]
    subject_ids = [x['id_str'] for x in subjects]
    labels = [x['labels'] for x in subjects]
    db_access.add_labels_to_db(subject_ids, labels, filled_shard_db)
    # read db, check labels match
    cursor = filled_shard_db.cursor()
    for subject in subjects:
        cursor.execute(
            '''
            SELECT labels FROM catalog
            WHERE id_str = (:id_str)
            ''',
            (subject['id_str'],)
        )
        results = list(cursor.fetchall())
        assert len(results) == 1
        assert results[0][0] == json.dumps(subject['labels'])


def test_get_all_entries(filled_shard_db_with_partial_labels):
    entries = db_access.get_all_entries(filled_shard_db_with_partial_labels)
    assert all([isinstance(x, db_access.CatalogEntry) for x in entries])
    assert [x.id_str for x in entries] == ['some_hash', 'some_other_hash', 'yet_another_hash']

def test_get_all_entries_labelled(filled_shard_db_with_partial_labels):
    entries = db_access.get_all_entries(filled_shard_db_with_partial_labels, labelled=True)
    assert all([isinstance(x, db_access.CatalogEntry) for x in entries])
    assert [x.id_str for x in entries] == ['some_hash']

def test_get_all_entries_unlabelled(filled_shard_db_with_partial_labels):
    entries = db_access.get_all_entries(filled_shard_db_with_partial_labels, labelled=False)
    assert all([isinstance(x, db_access.CatalogEntry) for x in entries])
    assert [x.id_str for x in entries] == ['some_other_hash', 'yet_another_hash']

def test_add_labels_to_db_already_labelled_should_fail(filled_shard_db_with_partial_labels):
    entries = [
        {
            'id_str': 'some_hash',
            'labels': json.dumps({'column': 1})
        }
    ]
    subject_ids = [x['id_str'] for x in entries]
    labels = [x['labels'] for x in entries]
    with pytest.raises(ValueError):
        db_access.add_labels_to_db(
            subject_ids, labels, filled_shard_db_with_partial_labels)

def test_get_all_shard_locs(filled_shard_db):
    assert db_access.get_all_shard_locs(filled_shard_db) == ['tfrecord_a', 'tfrecord_b']
