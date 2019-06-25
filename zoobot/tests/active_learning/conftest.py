import pytest

import sqlite3
import copy
import os
import time
import json
import hashlib

import numpy as np
import pandas as pd
from PIL import Image

from astropy.io import fits
from zoobot.active_learning import make_shards, create_instructions
from zoobot.tests import TEST_EXAMPLE_DIR


@pytest.fixture()
def n_subjects():
    return 256


@pytest.fixture()
def id_strs(n_subjects):
    return [str(n) for n in range(n_subjects)]


@pytest.fixture(params=['png_loc', 'fits_loc'])
def file_col(request):
    return request.param


@pytest.fixture
def catalog_random_images(size, channels, n_subjects, id_strs, fits_native_dir, png_native_dir, file_col):
    """Construct labelled/unlabelled catalogs for testing active learning"""
    
    
    assert os.path.isdir(fits_native_dir)
    assert os.path.isdir(png_native_dir)
    matrices = np.random.rand(n_subjects, size, size, channels)
    some_feature = np.random.rand(n_subjects)

    catalog = pd.DataFrame(data={'id_str': id_strs, 'some_feature': some_feature})

    if file_col == 'fits_loc':
        relative_fits_locs = ['random_{}.fits'.format(n) for n in range(n_subjects)]
        fits_locs = list(map(lambda rel_loc: os.path.join(fits_native_dir, rel_loc), relative_fits_locs))
        for matrix, loc in zip(matrices, fits_locs):  # write to fits
            hdu = fits.PrimaryHDU(matrix)
            hdu.writeto(loc, overwrite=True)
            assert os.path.isfile(loc)
        catalog['file_loc'] = fits_locs

    if file_col == 'png_loc':
        relative_png_locs = ['random_{}.png'.format(n) for n in range(n_subjects)]
        png_locs = list(map(lambda rel_loc: os.path.join(png_native_dir, rel_loc), relative_png_locs))
        for matrix, loc in zip(matrices, png_locs):  # write to fits
            rgb_matrix = (matrix * 256).astype(np.uint8)
            Image.fromarray(rgb_matrix, mode='RGB').save(loc)
            assert os.path.isfile(loc)
        catalog['file_loc'] = png_locs

    return catalog


# due to conftest.py, catalog fits_loc is absolute and points to fits_native_dir
@pytest.fixture()
def labelled_catalog(catalog_random_images):
    catalog = catalog_random_images.copy()
    catalog['id_str'] = catalog_random_images['id_str'] + '_from_labelled'  # must be unique
    catalog['label_a'] = np.random.rand(len(catalog))
    catalog['label_b'] = np.random.rand(len(catalog))
    catalog['total_votes'] = np.random.randint(low=1, high=41, size=len(catalog))
    return catalog


@pytest.fixture()
def unlabelled_catalog(catalog_random_images):
    catalog = catalog_random_images.copy()
    catalog['id_str'] = catalog_random_images['id_str'] + '_from_unlabelled'  # must be unique
    return catalog


@pytest.fixture()
def shard_config(tmpdir, size, channels):
    config = make_shards.ShardConfig(
        shard_dir=tmpdir.mkdir('base_dir').strpath,
        shard_size=128,
        inital_size=size,
        final_size=size,
        channels=channels)
    return config


@pytest.fixture()
def shard_config_ready(shard_config, labelled_catalog, unlabelled_catalog):
    config = copy.copy(shard_config)
    config.prepare_shards(labelled_catalog, unlabelled_catalog, train_test_fraction=0.8)
    assert config.ready()
    return config


@pytest.fixture()
def db_loc(tmpdir):
    return os.path.join(tmpdir.mkdir('db_dir').strpath, 'db_is_here.db')


def mock_acquisition_func(samples):
    assert isinstance(samples, np.ndarray)
    assert len(samples.shape) == 2
    return samples.mean(axis=1)  # sort by mean prediction (here, mean of each subject)


def mock_train_callable(estimators_dir, train_tfrecord_locs, eval_tfrecord_locs, learning_rate, epochs):
    # pretend to save a model in subdirectory of estimator_dir
    assert os.path.isdir(estimators_dir)
    subdir_loc = os.path.join(estimators_dir, str(time.time()))
    os.mkdir(subdir_loc)
    with open(os.path.join(subdir_loc, 'dummy_model.txt'), 'w') as f:
        json.dump(train_tfrecord_locs, f)


@pytest.fixture()
def acquisition():
    return np.random.rand()


@pytest.fixture()
def acquisitions(subjects):
    return list(np.random.rand(len(subjects)))


@pytest.fixture()
def images(n_subjects, size):
    return np.random.rand(n_subjects, size, size, 3)


@pytest.fixture()
def subjects(size, images):
    return [{'matrix': images[n], 'id_str': 'id_' + str(n)} for n in range(len(images))]


def mock_get_samples_of_images(model, images, n_samples):
    # predict the mean of image batch, 10 times
    assert isinstance(images, np.ndarray)
    assert len(images.shape) == 4
    assert isinstance(n_samples, int)
    response = [[np.mean(images[n])] * 10 for n in range(len(images))]
    return np.array(response)


@pytest.fixture()
def samples(images):
    return mock_get_samples_of_images(model=None, images=images, n_samples=10)


@pytest.fixture()
def estimators_dir(tmpdir):
    base_dir = tmpdir.mkdir('estimators').strpath
    checkpoint_dirs = ['157001', '157002', '157003']
    for directory in checkpoint_dirs:
        os.mkdir(os.path.join(base_dir, directory))
    files = ['checkpoint', 'graph.pbtxt', 'events.out.tfevents.1545', 'model.ckpt.2505.index', 'model.ckpt.2505.meta']
    for file_name in files:
        with open(os.path.join(base_dir, file_name), 'w') as f:
            f.write('Dummy file')
    return base_dir


@pytest.fixture()
def shard_config_loc(tmpdir):
    shard_dir = tmpdir.mkdir('shard_dir').strpath
    shard_config_loc = os.path.join(shard_dir, 'shard_config.json')
    shard_data = {
        'shard_dir': shard_dir,
        'train_dir': os.path.join(shard_dir, 'some_train_dir'),
        'eval_dir': os.path.join(shard_dir, 'some_eval_dir'),
        'labelled_catalog_loc': os.path.join(shard_dir, 'some_labelled_catalog_loc'),
        'unlabelled_catalog_loc': os.path.join(shard_dir, 'some_unlabelled_catalog_loc'),
        'config_save_loc': os.path.join(shard_dir, 'some_config_save_loc'),
        'db_loc': os.path.join(shard_dir, 'some_db_loc.db')
    }
    with open(shard_config_loc, 'w') as f:
        json.dump(shard_data, f)
    # with open(shard_data['db_loc'], 'w') as f:
    #     f.write('dummy shard db')
    return shard_config_loc


@pytest.fixture(params=[True, False])
def initial_estimator_ckpt(request, predictor_model_loc):
    if request.param:
        return predictor_model_loc
    else:
        return None


@pytest.fixture()
def instructions(mocker, shard_config_loc, tmpdir, predictor_model_loc, warm_start, initial_estimator_ckpt):

    # replace load_shard_config with a mock, so we don't need to actually make the shards
    mocker.patch(
        'zoobot.active_learning.create_instructions.make_shards.load_shard_config', 
        autospec=True
    )
    
    config = create_instructions.Instructions(
        shard_config_loc, 
        save_dir=tmpdir.mkdir('run_dir').strpath,
        shards_per_iter=2,
        subjects_per_iter=10,
        initial_estimator_ckpt=initial_estimator_ckpt,
        n_samples=15
        )

    assert os.path.isdir(config.save_dir)  # permanent directory for dvc control
    assert os.path.exists(config.db_loc)

    assert config.ready()
    return config


@pytest.fixture(params=[True, False])
def baseline(request):
    return request.param


@pytest.fixture(params=[True, False])
def test(request):
    return request.param


@pytest.fixture(params=[True, False])
def warm_start(request):
    return request.param



@pytest.fixture()
def unknown_subject(size, channels):
    return {
        'matrix': np.random.rand(size, size, channels),
        'id_str': hashlib.sha256(b'some_id_bytes').hexdigest()
    }


@pytest.fixture()
def known_subject(known_subject):
    known_subject = unknown_subject.copy()
    known_subject['label'] = np.random.randint(1)
    return known_subject


@pytest.fixture()
def test_dir(tmpdir):
    return tmpdir.strpath


@pytest.fixture(params=['fits', 'png'])
def file_loc_of_image(request):
    if request.param == 'fits':
        loc = os.path.join(TEST_EXAMPLE_DIR, 'example_a.fits')
    elif request.param == 'png':
        loc = os.path.join(TEST_EXAMPLE_DIR, 'example_a.png')
    else:
        raise ValueError(request.param)
    assert os.path.isfile(loc)
    return loc


@pytest.fixture()
def filled_shard_db(empty_shard_db, file_loc_of_image):
    # some_hash, some_other_hash and yet_another_hash are unlabelled but exist
    db = empty_shard_db
    cursor = db.cursor()

    cursor.execute(
        '''
        INSERT INTO catalog(id_str, file_loc)
                  VALUES(:id_str, :file_loc)
        ''',
        {
            'id_str': 'some_hash',
            'file_loc': file_loc_of_image
        }
    )
    db.commit()
    cursor.execute(
        '''
        INSERT INTO shards(id_str, tfrecord_loc)
                  VALUES(:id_str, :tfrecord_loc)
        ''',
        {
            'id_str': 'some_hash',
            'tfrecord_loc': 'tfrecord_a'
        }
    )
    db.commit()

    cursor.execute(
        '''
        INSERT INTO catalog(id_str, file_loc)
                  VALUES(:id_str, :file_loc)
        ''',
        {
            'id_str': 'some_other_hash',
            'file_loc': file_loc_of_image
        }
    )
    db.commit()
    cursor.execute(
        '''
        INSERT INTO shards(id_str, tfrecord_loc)
                  VALUES(:id_str, :tfrecord_loc)
        ''',
        {
            'id_str': 'some_other_hash',
            'tfrecord_loc': 'tfrecord_b'
        }
    )
    db.commit()

    cursor.execute(
        '''
        INSERT INTO catalog(id_str, file_loc)
                  VALUES(:id_str, :file_loc)
        ''',
        {
            'id_str': 'yet_another_hash',
            'file_loc': file_loc_of_image
        }
    )
    db.commit()
    cursor.execute(
        '''
        INSERT INTO shards(id_str, tfrecord_loc)
                  VALUES(:id_str, :tfrecord_loc)
        ''',
        {
            'id_str': 'yet_another_hash',
            'tfrecord_loc': 'tfrecord_a'  # same as first entry, should be selected if filter on rec a
        }
    )
    db.commit()
    return db


def make_db(db, rows):
    cursor = db.cursor()
    for row in rows:
        cursor.execute(
            '''
            UPDATE catalog SET labels = ?
            WHERE id_str = ?
            ''',
            (row['labels'], row['id_str'])
        )
        db.commit()
    return db


@pytest.fixture()
def filled_shard_db_with_partial_labels(filled_shard_db):
    # some_hash is labelled, some_other_hash and yet_another_hash are unlabelled but exist
    rows = [
        {
            'id_str': 'some_hash',
            'labels': json.dumps({'column': 1})
         },
         {
            'id_str': 'some_other_hash',
            'labels': None
        },
        {
            'id_str': 'yet_another_hash',
            'labels': None
        }
    ]
    db = make_db(filled_shard_db, rows)
    # trust but verify
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT id_str, labels FROM catalog
        '''
    )
    catalog = cursor.fetchall()
    assert catalog == [('some_hash', json.dumps({'column': 1})), ('some_other_hash', None), ('yet_another_hash', None)]
    return db


@pytest.fixture()
def filled_shard_db_with_labels(filled_shard_db):
    # all subjects labelled
    rows = [
        {
            'id_str': 'some_hash',
            'labels': json.dumps({'column': 1})
         },
         {
            'id_str': 'some_other_hash',
            'labels': json.dumps({'column': 1})
        },
        {
            'id_str': 'yet_another_hash',
            'labels': json.dumps({'column': 1})
        }
    ]
    db = make_db(filled_shard_db, rows)
    # trust but verify
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT id_str, labels FROM catalog
        '''
    )
    catalog = cursor.fetchall()
    assert catalog == [('some_hash', json.dumps({'column': 1})), ('some_other_hash', json.dumps({'column': 1})), ('yet_another_hash', json.dumps({'column': 1}))]
    return db

@pytest.fixture()
def empty_shard_db():
    db = sqlite3.connect(':memory:')

    cursor = db.cursor()

    cursor.execute(
        '''
        CREATE TABLE catalog(
            id_str STRING PRIMARY KEY,
            labels STRING DEFAULT NULL,
            file_loc STRING)
        '''
    )
    db.commit()

    cursor.execute(
        '''
        CREATE TABLE shards(
            id_str STRING PRIMARY KEY,
            tfrecord_loc TEXT)
        '''
    )
    db.commit()
    return db

