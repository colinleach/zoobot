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
from zoobot.active_learning import active_learning, default_estimator_params, setup
from zoobot.tests import TEST_EXAMPLE_DIR
from zoobot.tests import active_learning_test

ROOT = '/home/ubuntu'

class ShardConfig():
    # catalog, unlabelled shards, and single shard of labelled subjects
    # should be JSON serializable
    # at time of creation, many paths may not yet resolve - aimed at later run_dir
    def __init__(
        self,
        base_dir,  # to hold a new folder, named after the shard config 
        inital_size=128,
        final_size=96,
        shard_size=1024,
        label_split_value='0.4',
        **overflow_args
        ):

        self.base_dir = base_dir

        self.label_split_value = label_split_value
        self.initial_size = inital_size
        self.final_size = final_size
        self.channels = 3
        self.shard_size = shard_size

        self.run_name = 'shards_si{}_sf{}_l{}'.format(
            self.initial_size, 
            self.final_size, 
            self.label_split_value
        )

        self.shard_dir = os.path.join(base_dir, self.run_name)
        self.db_loc = os.path.join(self.shard_dir, 'static_shard_db.db')  # assumed

        self.train_tfrecord_loc = os.path.join(self.shard_dir, 'initial_train.tfrecord')
        self.eval_tfrecord_loc = os.path.join(self.shard_dir, 'eval.tfrecord')

        # catalog `fits_loc_relative` column is relative to this directory
        # holds all fits in both catalogs, used for writing shards
        # NOT copied to snapshot: fits are only copied as they are labelled
        self.ec2_fits_dir = os.path.join(ROOT, 'fits_native')

        self.labelled_catalog_loc = os.path.join(self.shard_dir, 'labelled_catalog.csv')
        self.unlabelled_catalog_loc = os.path.join(self.shard_dir, 'unlabelled_catalog.csv')

        self.config_save_loc = os.path.join(self.shard_dir, 'shard_config.json')


    def prepare_shards(self, labelled_catalog, unlabelled_catalog):
        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir)  # always fresh
        os.mkdir(self.shard_dir)

        # assume the catalog is true, don't modify halfway through!
        # labelled_catalog['fits_loc'] = self.ec2_fits_dir + labelled_catalog['fits_loc_relative']
        # unlabelled_catalog['fits_loc'] = self.ec2_fits_dir + unlabelled_catalog['fits_loc_relative']

        labelled_catalog.to_csv(self.labelled_catalog_loc)
        unlabelled_catalog.to_csv(self.unlabelled_catalog_loc)

        setup.make_database_and_shards(
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
            train_test_fraction=0.8)

        assert self.ready()


    def ready(self):
        assert os.path.isdir(self.shard_dir)
        assert os.path.isdir(self.ec2_fits_dir)
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
        with open(self.config_save_loc, 'w') as f:
            json.dump(self.to_dict(), f)



def snapshot_shards(volume_base_dir, catalog_loc):
    # in memory for now, but will be serialized for later/logs
    shard_config = ShardConfig(base_dir=volume_base_dir)  

    # in memory for now, but will be saved to csv
    catalog = pd.read_csv(catalog_loc)
    # >36 votes required, gives low count uncertainty
    catalog = catalog[catalog['smooth-or-featured_total-votes'] > 36]
    catalog['label'] = (catalog['smooth-or-featured_smooth_fraction'] > float(shard_config.label_split_value)).astype(int)  # 0 for featured
    catalog['id_str'] = catalog['subject_id'].astype(str) 

    # temporary hacks for mocking panoptes
    # save catalog for mock_panoptes.py to return (now added to git)
    # catalog[['id_str', 'label']].to_csv(os.path.join(TEST_EXAMPLE_DIR, 'mock_panoptes.csv'), index=False)
    # split catalog and pretend most is unlabelled
    labelled_catalog = catalog[:1024]  # for initial training data
    unlabelled_catalog = catalog[1024:1024*7]  # for new data
    del unlabelled_catalog['label']


    shard_config.prepare_shards(
        labelled_catalog,
        unlabelled_catalog)

    # must be able to end here, snapshot created and ready to go (hopefully)
    shard_config.write()


if __name__ == '__main__':

    logging.basicConfig(
        filename='make_shards_{}.log'.format(time.time()),
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )


    parser = argparse.ArgumentParser(description='Make shards')
    parser.add_argument('--base_dir', dest='base_dir', type=str,
                    help='Directory into which to place shard directory')
    parser.add_argument('--catalog_loc', dest='catalog_loc', type=str,
                    help='Path to csv catalog of Panoptes labels and fits_loc, for shards')
    args = parser.parse_args()

    # laptop_base = '/users/mikewalmsley/pretend_ec2_root'
    # ec2_base = '/home/ec2-user'
    # ec2_base = '/home/ubuntu'

    # laptop_catalog_loc = '/users/mikewalmsley/repos/zoobot/zoobot/tests/test_examples/panoptes_predictions.csv'
    # ec2_catalog_loc = ec2_base + '/panoptes_predictions.csv'

    # laptop_shard_loc = '/users/mikewalmsley/pretend_ec2_root/'
    # ec2_shard_loc = ec2_base + '/shards_si64_sf28_l0.4/shard_config.json'

    # laptop_run_dir_baseline = '/users/mikewalmsley/pretend_ec2_root/run_baseline'
    # ec2_run_dir_baseline = ec2_base + '/run_baseline'
    # ec2_run_dir = ec2_base + '/run'
    
    snapshot_shards(
        volume_base_dir=args.base_dir,
        catalog_loc=args.catalog_loc)
