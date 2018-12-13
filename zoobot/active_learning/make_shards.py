import argparse
import os
import shutil
import logging
import json
import time

import numpy as np
import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params
from zoobot.tests import TEST_EXAMPLE_DIR


class ShardConfig():
    """
    Assumes that you have:
    - a directory of fits files  (e.g. `fits_native`)
    - a catalog of files under 'fits_loc'

    - a catalog with column `fits_loc_relative` pointing to those files, relative to that directory

    Checks that catalog paths match real fits files
    Creates unlabelled shards and single shard of labelled subjects
    Creates sqlite database describing what's in those shards

    JSON serializable for later loading
    """

    def __init__(
        self,
        shard_dir,  # to hold a new folder, named after the shard config 
        inital_size=128,
        final_size=64,
        shard_size=4096,
        **overflow_args
        ):

        self.initial_size = inital_size
        self.final_size = final_size
        self.channels = 3
        self.shard_size = shard_size
        self.shard_dir = shard_dir
        self.db_loc = os.path.join(self.shard_dir, 'static_shard_db.db')  # assumed

        self.train_tfrecord_loc = os.path.join(self.shard_dir, 'initial_train.tfrecord')
        self.eval_tfrecord_loc = os.path.join(self.shard_dir, 'eval.tfrecord')

        self.labelled_catalog_loc = os.path.join(self.shard_dir, 'labelled_catalog.csv')
        self.unlabelled_catalog_loc = os.path.join(self.shard_dir, 'unlabelled_catalog.csv')

        self.config_save_loc = os.path.join(self.shard_dir, 'shard_config.json')



    def prepare_shards(self, labelled_catalog, unlabelled_catalog):
        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir)  # always fresh
        os.mkdir(self.shard_dir)

        # check that fits_loc columns resolve correctly
        assert os.path.isfile(labelled_catalog['fits_loc'].iloc[0])
        assert os.path.isfile(unlabelled_catalog['fits_loc'].iloc[0])

        # assume the catalog is true, don't modify halfway through
        labelled_catalog.to_csv(self.labelled_catalog_loc)
        unlabelled_catalog.to_csv(self.unlabelled_catalog_loc)

        make_database_and_shards(
            unlabelled_catalog, 
            self.db_loc, 
            self.initial_size, 
            self.shard_dir, 
            self.shard_size)

        catalog_to_tfrecord.write_catalog_to_train_test_tfrecords(
            labelled_catalog, 
            self.train_tfrecord_loc, 
            self.eval_tfrecord_loc, 
            self.initial_size, 
            ['id_str', 'label'], 
            train_test_fraction=0.3)  # 30% train, 70% test

        assert self.ready()

        # serialized for later/logs
        self.write()


    def ready(self):
        assert os.path.isdir(self.shard_dir)
        # assert os.path.isdir(self.fits_dir)
        assert os.path.isfile(self.train_tfrecord_loc)
        assert os.path.isfile(self.eval_tfrecord_loc)
        assert os.path.isfile(self.db_loc)
        assert os.path.isfile(self.labelled_catalog_loc)
        assert os.path.isfile(self.unlabelled_catalog_loc)
        return True


    # TODO move to shared utilities
    def to_dict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])

    
    def write(self):
        with open(self.config_save_loc, 'w+') as f:
            json.dump(self.to_dict(), f)


def load_shard_config(shard_config_loc):
    with open(shard_config_loc, 'r') as f:
        shard_config_dict = json.load(f)
    return ShardConfig(**shard_config_dict)


def make_database_and_shards(catalog, db_loc, size, shard_dir, shard_size):
    if os.path.exists(db_loc):
        os.remove(db_loc)
    # set up db and shards using unknown catalog data
    db = active_learning.create_db(catalog, db_loc)
    columns_to_save = ['id_str']
    active_learning.write_catalog_to_tfrecord_shards(catalog, db, size, columns_to_save, shard_dir, shard_size=shard_size)


if __name__ == '__main__':

    # Write catalog to shards (tfrecords as catalog chunks) for use in active learning
    parser = argparse.ArgumentParser(description='Make shards')
    parser.add_argument('--shard_dir', dest='shard_dir', type=str,
                    help='Directory into which to place shard directory')
    parser.add_argument('--catalog_loc', dest='catalog_loc', type=str,
                    help='Path to csv catalog of Panoptes labels and fits_loc, for shards')
    args = parser.parse_args()

    logging.basicConfig(
        filename='make_shards_{}.log'.format(time.time()),
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )

    # in memory for now, but will be saved to csv
    catalog = pd.read_csv(args.catalog_loc)
    # >36 votes required, gives low count uncertainty
    catalog = catalog[catalog['smooth-or-featured_total-votes'] > 36]
    catalog['label'] = catalog['smooth-or-featured_smooth_fraction']  # float, 0. for featured
    catalog['id_str'] = catalog['subject_id'].astype(str) 

    # temporary hacks for mocking panoptes
    # save catalog for mock_panoptes.py to return (now added to git)
    catalog[['id_str', 'label']].to_csv(os.path.join(TEST_EXAMPLE_DIR, 'mock_panoptes.csv'), index=False)
    # split catalog and pretend most is unlabelled
    labelled_catalog = catalog[:1024]  # for initial training data
    unlabelled_catalog = catalog[1024:]  # for new data
    del unlabelled_catalog['label']

    # in memory for now, but will be serialized for later/logs
    shard_config = ShardConfig(shard_dir=args.shard_dir)  
    shard_config.prepare_shards(
        labelled_catalog,
        unlabelled_catalog)
    # must be able to end here, snapshot created and ready to go (hopefully)
