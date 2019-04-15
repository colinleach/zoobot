"""No knowledge of oracle allowed!
"""
import argparse
import os
import shutil
import logging
import json
import time

import numpy as np
import pandas as pd
import git

from shared_astro_utils import matching_utils, object_utils

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params, mock_panoptes, prepare_catalogs
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
        size=256,  # IMPORTANT
        shard_size=4096,
        **overflow_args  # TODO review removing this
        ):
        """
        Args:
            shard_dir (str): directory into which to save shards
            size (int, optional): Defaults to 128. Resolution to save fits to tfrecord
            final_size (int, optional): Defaults to 64. Resolution to load from tfrecord into model
            shard_size (int, optional): Defaults to 4096. Galaxies per shard.
        """
        self.size = size
        self.shard_size = shard_size
        self.shard_dir = shard_dir

        self.channels = 3  # save 3-band image to tfrecord. Augmented later by model input func.

        self.db_loc = os.path.join(self.shard_dir, 'static_shard_db.db')  # record shard contents

        # paths for fixed tfrecords for initial training and (permanent) evaluation
        self.train_dir = os.path.join(self.shard_dir, 'train_shards') 
        self.eval_dir = os.path.join(self.shard_dir, 'eval_shards')

        # paths for catalogs. Used to look up .fits locations during active learning.
        self.labelled_catalog_loc = os.path.join(self.shard_dir, 'labelled_catalog.csv')
        self.unlabelled_catalog_loc = os.path.join(self.shard_dir, 'unlabelled_catalog.csv')

        self.config_save_loc = os.path.join(self.shard_dir, 'shard_config.json')


    def train_tfrecord_locs(self):
        return [os.path.join(self.train_dir, loc) for loc in os.listdir(self.train_dir)
            if loc.endswith('.tfrecord')]


    def eval_tfrecord_locs(self):
        return [os.path.join(self.eval_dir, loc) for loc in os.listdir(self.eval_dir)
            if loc.endswith('.tfrecord')]


    def prepare_shards(self, labelled_catalog, unlabelled_catalog, train_test_fraction):
        """[summary]
        
        Args:
            labelled_catalog (pd.DataFrame): labelled galaxies, including fits_loc column
            unlabelled_catalog (pd.DataFrame): unlabelled galaxies, including fits_loc column
            train_test_fraction (float): fraction of labelled catalog to use as training data
        """
        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir)  # always fresh
        os.mkdir(self.shard_dir)
        os.mkdir(self.train_dir)
        os.mkdir(self.eval_dir)

        # check that file paths resolve correctly
        print(labelled_catalog['file_loc'][:3].values)
        prepare_catalogs.check_no_missing_files(labelled_catalog['file_loc'])
        prepare_catalogs.check_no_missing_files(unlabelled_catalog['file_loc'])

        # assume the catalog is true, don't modify halfway through
        logging.info('\nLabelled subjects: {}'.format(len(labelled_catalog)))
        logging.info('Unlabelled subjects: {}'.format(len(unlabelled_catalog)))
        labelled_catalog.to_csv(self.labelled_catalog_loc)
        unlabelled_catalog.to_csv(self.unlabelled_catalog_loc)

        # save train/test split into training and eval shards
        train_df, eval_df = catalog_to_tfrecord.split_df(labelled_catalog, train_test_fraction=train_test_fraction)
        logging.info('\nTraining subjects: {}'.format(len(train_df)))
        logging.info('Eval subjects: {}'.format(len(eval_df)))
        if len(train_df) < len(eval_df):
            print('More eval subjects than training subjects - is this intended?')
        train_df.to_csv(os.path.join(self.train_dir, 'train_df.csv'))
        eval_df.to_csv(os.path.join(self.eval_dir, 'eval_df.csv'))

        for (df, save_dir) in [(train_df, self.train_dir), (eval_df, self.eval_dir)]:
            active_learning.write_catalog_to_tfrecord_shards(
                df,
                db=None,
                img_size=self.size,
                columns_to_save=['id_str', 'label', 'total_votes'],
                save_dir=save_dir,
                shard_size=self.shard_size
            )

        make_database_and_shards(
            unlabelled_catalog, 
            self.db_loc, 
            self.size, 
            self.shard_dir, 
            self.shard_size)

        assert self.ready()

        # serialized for later/logs
        self.write()


    def ready(self):
        assert os.path.isdir(self.shard_dir)
        assert os.path.isdir(self.train_dir)
        assert os.path.isdir(self.eval_dir)
        assert os.path.isfile(self.db_loc)
        assert os.path.isfile(self.labelled_catalog_loc)
        assert os.path.isfile(self.unlabelled_catalog_loc)
        return True


    def to_dict(self):
        return object_utils.object_to_dict(self)

    def write(self):
        with open(self.config_save_loc, 'w+') as f:
            json.dump(self.to_dict(), f)


def load_shard_config(shard_config_loc):
    # shards to use
    shard_config = load_shard_config_naive(shard_config_loc)
    # update shard paths in case shard dir was moved since creation
    new_shard_dir = os.path.dirname(shard_config_loc)
    shard_config.shard_dir = new_shard_dir
    attrs = [
        'train_dir',
        'eval_dir',
        'labelled_catalog_loc',
        'unlabelled_catalog_loc',
        'config_save_loc',
        'db_loc']
    for attr in attrs:
        old_loc = getattr(shard_config, attr)
        new_loc = os.path.join(new_shard_dir, os.path.split(old_loc)[-1])
        logging.info('Was {}, now {}'.format(attr, new_loc))
        setattr(shard_config, attr, new_loc)
    return shard_config


def load_shard_config_naive(shard_config_loc):
    with open(shard_config_loc, 'r') as f:
        shard_config_dict = json.load(f)
    return ShardConfig(**shard_config_dict)



def make_database_and_shards(catalog, db_loc, size, shard_dir, shard_size):
    if os.path.exists(db_loc):
        os.remove(db_loc)
    # set up db and shards using unknown catalog data
    db = active_learning.create_db(catalog, db_loc)
    columns_to_save = ['id_str']
    active_learning.write_catalog_to_tfrecord_shards(
        catalog,
        db,
        size,
        columns_to_save,
        shard_dir,
        shard_size
    )



if __name__ == '__main__':

    # only responsible for making the shards. 
    # Assumes catalogs are shuffled and have id_str, file_loc, label, total_votes columns

    parser = argparse.ArgumentParser(description='Make shards')
    # to create for GZ2, see zoobot/get_catalogs/gz2
    # to create for DECALS, see github/zooniverse/decals and apply zoobot/active_learning/make_decals_catalog to `joint_catalog_for_upload`
    parser.add_argument('--labelled-catalog', dest='labelled_catalog_loc', type=str,
                    help='Path to csv catalog of previous labels and file_loc, for shards')

    parser.add_argument('--unlabelled-catalog', dest='unlabelled_catalog_loc', type=str,
                help='Path to csv catalog of previous labels and file_loc, for shards')

    parser.add_argument('--eval-size', dest='eval_size', type=str,
            help='Path to csv catalog of previous labels and file_loc, for shards')

    # Write catalog to shards (tfrecords as catalog chunks) here for use in active learning
    parser.add_argument('--shard-dir', dest='shard_dir', type=str,
                    help='Directory into which to place shard directory')

    args = parser.parse_args()

    log_loc = 'make_shards_{}.log'.format(time.time())
    logging.basicConfig(
        filename=log_loc,
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )

    # TODO folder strategy is a bit messed up before/after shards
    labelled_catalog = pd.read_csv(args.labelled_catalog_loc)
    unlabelled_catalog = pd.read_csv(args.unlabelled_catalog_loc)
    logging.info('Labelled: {}, unlabelled: {}'.format(len(labelled_catalog), len(unlabelled_catalog)))

    # in memory for now, but will be serialized for later/logs
    train_test_fraction = (len(labelled_catalog) - int(args.eval_size))/len(labelled_catalog)  # always eval on random 2500 galaxies
    logging.info('Train test fraction: {}'.format(train_test_fraction))

    shard_config = ShardConfig(shard_dir=args.shard_dir)

    shard_config.prepare_shards(
        labelled_catalog,
        unlabelled_catalog,
        train_test_fraction=train_test_fraction
    )
    
    # finally, tidy up by moving the log into the shard directory
    # could not be create here because shard directory did not exist at start of script
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    shutil.move(log_loc, os.path.join(args.shard_dir, '{}.log'.format(sha)))
