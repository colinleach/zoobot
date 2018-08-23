import pytest

import logging
import os
import random
import hashlib
import sqlite3

import numpy as np
import tensorflow as tf
import pandas as pd

from zoobot.tfrecord import create_tfrecord, read_tfrecord
from zoobot.estimators.estimator_params import default_four_layer_architecture, default_params
from zoobot.active_learning import active_learning


logging.basicConfig(
    filename='active_learning_test.log',
    filemode='w',
    format='%(asctime)s %(message)s',
    level=logging.DEBUG)


@pytest.fixture()
def unknown_subject(size, channels):
    return {
        'matrix': np.random.rand(size, size, channels),
        'id': hashlib.sha256(b'some_id_bytes').hexdigest()
    }


@pytest.fixture()
def known_subject(known_subject):
    known_subject = unknown_subject.copy()
    known_subject['label'] = np.random.randint(1)
    return known_subject


@pytest.fixture()
def test_dir(tmpdir):
    return tmpdir.strpath


@pytest.fixture()
def empty_shard_db():
    db = sqlite3.connect(':memory:')

    cursor = db.cursor()

    cursor.execute(
        '''
        CREATE TABLE catalog(
            id STRING PRIMARY KEY,
            label INT)
        '''
    )
    db.commit()

    cursor.execute(
        '''
        CREATE TABLE shardindex(
            id STRING PRIMARY KEY,
            tfrecord TEXT)
        '''
    )
    db.commit()

    cursor.execute(
        '''
        CREATE TABLE acquisitions(
            id STRING PRIMARY KEY,
            acquisition_value FLOAT)
        '''
    )
    db.commit()
    return db



@pytest.fixture()
def filled_shard_db(empty_shard_db):  # no shard index yet
    db = empty_shard_db
    cursor = db.cursor()
    cursor.execute(
        '''
        INSERT INTO acquisitions(id, acquisition_value)
                  VALUES(:id, :acquisition_value)
        ''',
        {
            'id': 'some_hash',
            'acquisition_value': 0.9
        }
    )
    db.commit()

    cursor.execute(
        '''
        INSERT INTO acquisitions(id, acquisition_value)
                  VALUES(:id, :acquisition_value)
        ''',
        {
            'id': 'some_other_hash',
            'acquisition_value': 0.3
        }
    )
    db.commit()

    return db


@pytest.fixture()
def shard_locs(stratified_tfrecord_locs):
    return stratified_tfrecord_locs  # pair of records. Bad - has label, does not have id


@pytest.fixture()
def acquisition_func():
    # return make_predictions.entropy
    return lambda x: np.random.rand(x.shape[0])


@pytest.fixture()
def acquisition():
    return np.random.rand()


def test_write_catalog_to_tfrecord_shards(catalog, empty_shard_db, size, channels, label_col, id_col, columns_to_save, tfrecord_dir):
    active_learning.write_catalog_to_tfrecord_shards(catalog, empty_shard_db, size, label_col, id_col, columns_to_save, tfrecord_dir, shard_size=10)
    # db should contain the catalog in 'catalog' table
    cursor = empty_shard_db.cursor()
    cursor.execute(
        '''
        SELECT id, label FROM catalog
        '''
    )
    catalog_entries = cursor.fetchall()
    for entry in catalog_entries:
        recovered_id = str(entry[0])
        recovered_label = entry[1]
        expected_label = catalog[catalog['id'] == recovered_id].squeeze()[label_col]
        assert recovered_label == expected_label

    # db should contain file locs in 'shardindex' table
    # tfrecords should have been written with the right files
    cursor.execute(
        '''
        SELECT id, tfrecord FROM shardindex
        '''
    )
    shardindex_entries = cursor.fetchall()
    shardindex_data = []
    for entry in shardindex_entries:
        shardindex_data.append({
            'id': str(entry[0]).encode('utf8'),  # TODO shardindex id is a byte string
            'tfrecord': entry[1]
        })
    shardindex = pd.DataFrame(data=shardindex_data)

    tfrecord_locs = shardindex['tfrecord'].unique()
    for tfrecord_loc in tfrecord_locs:
        expected_shard_ids = set(shardindex[shardindex['tfrecord'] == tfrecord_loc]['id'].unique())
        examples = read_tfrecord.load_examples_from_tfrecord(
            [tfrecord_loc], 
            read_tfrecord.matrix_label_id_feature_spec(size, channels)
        )
        actual_shard_ids = set([example['id'] for example in examples])
        assert expected_shard_ids == actual_shard_ids


def test_add_tfrecord_to_db(example_tfrecord_loc, empty_shard_db, catalog, id_col):  # bad loc
    active_learning.add_tfrecord_to_db(example_tfrecord_loc, empty_shard_db, catalog, id_col)
    cursor = empty_shard_db.cursor()
    cursor.execute(
        '''
        SELECT id, tfrecord FROM shardindex
        '''
    )
    saved_subjects = cursor.fetchall()
    for n, subject in enumerate(saved_subjects):
        assert str(subject[0]) == catalog.iloc[n][id_col]  # strange string casting when read back
        assert subject[1] == example_tfrecord_loc

def test_save_acquisition_to_db(unknown_subject, acquisition, empty_shard_db):
    active_learning.save_acquisition_to_db(unknown_subject, acquisition, empty_shard_db)
    cursor = empty_shard_db.cursor()
    cursor.execute(
        '''
        SELECT id, acquisition_value FROM acquisitions
        '''
    )
    saved_subject = cursor.fetchone()
    assert saved_subject[0] == unknown_subject['id']
    assert np.isclose(saved_subject[1], acquisition)


def test_record_acquisition_on_unlabelled(filled_shard_db, predictor, acquisition_func, shard_locs, size, channels):
    active_learning.record_acquisition_on_unlabelled(filled_shard_db, predictor, shard_locs, size, channels, acquisition_func, n_samples=10)
    cursor = filled_shard_db.cursor()
    cursor.execute(
        '''
        SELECT id, acquisition_value FROM acquisitions
        '''
    )
    saved_subjects = cursor.fetchall()
    for subject in saved_subjects:
        assert 0. > subject[1] < 1.  # doesn't actually verify value is consistent


@pytest.fixture()
def test_add_top_acquisitions_to_tfrecord(db):  # TODO optional selected shard(s)
    pass


# @pytest.fixture()
# def params(test_dir):
#     params = active_learning.get_active_learning_params()
#     params['known_tfrecord_loc'] = test_dir + 'known.tfrecord'
#     params['unknown_tfrecord_loc'] = test_dir + 'unknown.tfrecord'


# def test_run_experiment(estimator, params, known_subjects, unknown_subjects):
#     active_learning.run_active_learning(estimator, params, known_subjects, unknown_subjects)
#     assert os.path.exists(params['log_dir'])
