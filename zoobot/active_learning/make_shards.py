import argparse
import os
import shutil
import logging
import json
import time

import numpy as np
import pandas as pd
import git

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params
from zoobot.tests import TEST_EXAMPLE_DIR


class ShardConfig():
    """
    Assumes that you have:
    - a directory of fits files  (e.g. `fits_native`)
    - a catalog of files, with file locations under the column 'fits_loc' (relative to repo root)

    Checks that catalog paths match real fits files
    Creates unlabelled shards and single shard of labelled subjects
    Creates sqlite database describing what's in those shards

    JSON serializable for later loading
    """

    def __init__(
        self,
        shard_dir,  # to hold a new folder, named after the shard config 
        inital_size=256,
        final_size=128,  # TODO consider refactoring this into execute.py
        shard_size=4096,
        **overflow_args  # TODO review removing this
        ):
        """
        Args:
            shard_dir (str): directory into which to save shards
            inital_size (int, optional): Defaults to 128. Resolution to save fits to tfrecord
            final_size (int, optional): Defaults to 64. Resolution to load from tfrecord into model
            shard_size (int, optional): Defaults to 4096. Galaxies per shard.
        """
        self.initial_size = inital_size
        self.final_size = final_size
        self.shard_size = shard_size
        self.shard_dir = shard_dir

        self.channels = 3  # save 3-band image to tfrecord. Augmented later by model input func.

        self.db_loc = os.path.join(self.shard_dir, 'static_shard_db.db')  # record shard contents

        # paths for fixed tfrecords for initial training and (permanent) evaluation
        self.train_tfrecord_loc = os.path.join(self.shard_dir, 'initial_train.tfrecord') 
        self.eval_tfrecord_loc = os.path.join(self.shard_dir, 'eval.tfrecord')

        # paths for catalogs. Used to look up .fits locations during active learning.
        self.labelled_catalog_loc = os.path.join(self.shard_dir, 'labelled_catalog.csv')
        self.unlabelled_catalog_loc = os.path.join(self.shard_dir, 'unlabelled_catalog.csv')

        self.config_save_loc = os.path.join(self.shard_dir, 'shard_config.json')


    def prepare_shards(self, labelled_catalog, unlabelled_catalog, train_test_fraction=0.1):
        """[summary]
        
        Args:
            labelled_catalog (pd.DataFrame): labelled galaxies, including fits_loc column
            unlabelled_catalog (pd.DataFrame): unlabelled galaxies, including fits_loc column
            train_test_fraction (float): fraction of labelled catalog to use as training data
        """

        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir)  # always fresh
        os.mkdir(self.shard_dir)

        # check that fits_loc columns resolve correctly
        assert os.path.isfile(labelled_catalog['png_loc'].iloc[0])
        assert os.path.isfile(unlabelled_catalog['png_loc'].iloc[0])

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
            train_test_fraction=train_test_fraction,
            source='png')

        assert self.ready()

        # serialized for later/logs
        self.write()


    def ready(self):
        assert os.path.isdir(self.shard_dir)
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
    # parser.add_argument('--catalog_loc', dest='catalog_loc', type=str,
    #                 help='Path to csv catalog of Panoptes labels and fits_loc, for shards')
    args = parser.parse_args()

    log_loc = 'make_shards_{}.log'.format(time.time())
    logging.basicConfig(
        filename=log_loc,
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )


    # needs update
    columns_to_save = [
        't01_smooth_or_features_a01_smooth_count',
        't01_smooth_or_features_a01_smooth_weighted_fraction',  # annoyingly, I only saved the weighted fractions. Should be quite similar, I hope.
        't01_smooth_or_features_a02_features_or_disk_count',
        't01_smooth_or_features_a03_star_or_artifact_count',
        'id',
        'ra',
        'dec'
    ]

    
    catalog_loc = '/data/galaxy_zoo/gz2/catalogs/basic_regression_labels_downloaded.csv'

    # only exists if zoobot/get_catalogs/gz2 instructions have been followed
    catalog = pd.read_csv(catalog_loc,
                        usecols=columns_to_save + ['png_loc', 'png_ready'],
                        nrows=None)


    # # in memory for now, but will be saved to csv
    # catalog = pd.read_csv(args.catalog_loc, usecols=columns_to_save)


    # 40 votes required, for accurate binomial statistics
    # catalog = catalog[catalog['smooth-or-featured_total-votes'] > 36]
    # catalog['label'] = catalog['smooth-or-featured_smooth_fraction']  # float, 0. for featured

    # previous catalog didn't include total classifications/votes, so we'll need to work around that for now
    catalog['smooth-or-featured_total-votes'] = catalog['t01_smooth_or_features_a01_smooth_count'] + catalog['t01_smooth_or_features_a02_features_or_disk_count'] + catalog['t01_smooth_or_features_a03_star_or_artifact_count']
    catalog = catalog[catalog['smooth-or-featured_total-votes'] > 36]  # >36 votes required, gives low count uncertainty

    # for consistency
    catalog['id_str'] = catalog['id'].astype(str)

    catalog['label'] = catalog['t01_smooth_or_features_a01_smooth_weighted_fraction']

    # catalog['id_str'] = catalog['subject_id'].astype(str)  # useful to crossmatch later

    # temporary hacks for mocking panoptes
    # save catalog for mock_panoptes.py to return (now added to git)
    # TODO a bit hacky, as only coincidentally the same
    dir_of_this_file = os.path.dirname(os.path.realpath(__file__))
    catalog[['id_str', 'label']].to_csv(os.path.join(dir_of_this_file, 'oracle_gz2.csv'), index=False)

    # with basic split, we do 80% train/test split
    # here, use 80% also but with 5*1024 pool held back as oracle (should be big enough)
    # select 1024 new training images
    # verify model is nearly as good as basic split (only missing about 4k images)
    # verify that can add thAese images to training pool without breaking everything!
    # may need to disable interleave, and instead make dataset of joined tfrecords (starting with new ones?)

    # print(len(catalog))
    # exit(0)

    # of 18k (exactly 40 votes), initial train on 6k, eval on 3k, and pool the remaining 9k
    # split catalog and pretend most is unlabelled
    # pool_size = 5*1024
    labelled_size = 5000

    labelled_catalog = catalog[:labelled_size]  # for training and eval. Could do basic split on these!
    unlabelled_catalog = catalog[labelled_size:10000]  # for pool
    del unlabelled_catalog['label']

    # in memory for now, but will be serialized for later/logs
    shard_config = ShardConfig(shard_dir=args.shard_dir)  
    shard_config.prepare_shards(
        labelled_catalog,
        unlabelled_catalog,
        train_test_fraction=0.8)  # copying basic_split
    # must be able to end here, snapshot created and ready to go (hopefully)

    # finally, tidy up by moving the log into the shard directory
    # could not be create here because shard directory did not exist at start of script
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    shutil.move(log_loc, os.path.join(args.shard_dir, '{}.log'.format(sha)))
