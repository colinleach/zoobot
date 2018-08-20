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
def known_subjects():
    data = [{
        'some_feature': np.random.rand(1),
        'label': np.random.randint(0, 2, size=1),
        'id': hashlib.sha256(b'some_id_bytes')
    }]
    return pd.DataFrame(data)


@pytest.fixture()
def unknown_subjects():
    data = [{
        'some_feature': np.random.rand(1),
        'id': hashlib.sha256(b'some_id_bytes')
    }]
    return pd.DataFrame(data)


@pytest.fixture()
def test_dir(tmpdir):
    return tmpdir.strpath


@pytest.fixture()
def empty_shard_db():
    db = sqlite3.connect(':memory:')

    cursor = db.cursor()
    cursor.execute
    ('''
    CREATE TABLE shard_index(
        id INTEGER PRIMARY KEY,
        saved_size INTEGER,
        tfrecord TEXT)
    ''')
    db.commit()

    cursor.execute
    ('''
    CREATE TABLE acquisitions(
        id INTEGER PRIMARY KEY,
        acquisition_value FLOAT)
    ''')
    db.commit()
    return db


@pytest.fixture()
def shard_locs(stratified_tfrecord_locs):
    return stratified_tfrecord_locs  # pair of records


@pytest.fixture()
def acquisition_func():
    # return make_predictions.entropy
    return lambda x: np.random.rand(x.shape[0])


@pytest.fixture()
def acquisition():
    return np.random.rand()


def test_write_catalog_to_tfrecord_shards(catalog, empty_shard_db, size, label_col, id_col, columns_to_save, tfrecord_dir):
    active_learning.write_catalog_to_tfrecord_shards(catalog, empty_shard_db, size, label_col, id_col, columns_to_save, tfrecord_dir, shard_size=10)


def test_add_tfrecord_to_db(tfrecord_loc, empty_shard_db):  # bad loc
    active_learning.add_tfrecord_to_db(tfrecord_loc, empty_shard_db)


def test_record_acquisition_on_unlabelled(empty_shard_db, predictor, shard_locs, size, channels, acquisition_func):
    active_learning.record_acquisition_on_unlabelled(empty_shard_db, predictor, shard_locs, size, channels, acquisition_func, n_samples=10)


def test_save_acquisition_to_db(subject, acquisition, empty_shard_db):
    active_learning.save_acquisition_to_db(subject, acquisition, empty_shard_db)


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
