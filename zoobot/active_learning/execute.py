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
from zoobot.active_learning import active_learning, default_estimator_params, setup, make_shards
from zoobot.tests import TEST_EXAMPLE_DIR
from zoobot.tests import active_learning_test


class ActiveConfig():

    def __init__(self, shard_config, run_dir):
        self.shards = shard_config
        self.run_dir = run_dir
        self.db_loc = os.path.join(self.run_dir, 'run_db.db')  
        self.estimator_dir = os.path.join(self.run_dir, 'estimator')

        # will download/copy fits of top acquisitions into here
        self.requested_fits_dir = os.path.join(self.run_dir, 'requested_fits')
        # and then write them into tfrecords here
        self.requested_tfrecords_dir = os.path.join(self.run_dir, 'requested_tfrecords')
        self.train_records_index_loc = os.path.join(self.run_dir, 'requested_tfrecords_index.json')

        self.max_iterations = 12
        self.n_subjects_per_iter = 512


    def prepare_run_folders(self):
        # new predictors (apart from initial disk load) for now
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)
        os.mkdir(self.run_dir)
        os.mkdir(self.estimator_dir)
        os.mkdir(self.requested_fits_dir)
        os.mkdir(self.requested_tfrecords_dir)

        shutil.copyfile(self.shards.db_loc, self.db_loc)  # copy initial shard db to here, to modify


    def ready(self):
        assert self.shards.ready()
        # TODO more validation checks for the run
        return True



def execute_active_learning(shard_config_loc, run_dir, baseline=False):
    # on another machine, at another time...
    with open(shard_config_loc, 'r') as f:
        shard_config_dict = json.load(f)
    shard_config = make_shards.ShardConfig(**shard_config_dict)
    active_config = ActiveConfig(shard_config, run_dir)
    active_config.prepare_run_folders()

    # define the estimator - load settings (rename 'setup' to 'settings'?)
    run_config = default_estimator_params.get_run_config(active_config)
    def train_callable(train_records):
        run_config.train_config.tfrecord_loc = train_records
        return run_estimator.run_estimator(run_config)

    if baseline:
        def mock_acquisition_func(predictor):  # predictor does nothing
            def mock_acquisition_callable(matrix_list):
                assert isinstance(matrix_list, list)
                assert all([isinstance(x, np.ndarray) for x in matrix_list])
                assert all([x.shape[0] == x.shape[1] for x in matrix_list])
                return [float(np.random.rand()) for x in matrix_list]
            return mock_acquisition_callable
        logging.warning('Using mock acquisition function, baseline test mode')
        get_acquisition_func = lambda predictor: mock_acquisition_func(predictor)  # random
    else:  # callable expecting predictor, returning a callable expecting matrix list
        get_acquisition_func = lambda predictor: make_predictions.get_acquisition_func(predictor, n_samples=20)

    unlabelled_catalog = pd.read_csv(
        active_config.shards.unlabelled_catalog_loc, 
        dtype={'id_str': str, 'label': int}
    )
    active_learning.run(
        unlabelled_catalog, 
        active_config.db_loc, 
        active_config.shards.initial_size, 
        3,  # TODO channels not really generalized
        active_config.estimator_dir, 
        active_config.shards.train_tfrecord_loc, 
        train_callable, 
        get_acquisition_func,
        active_config.max_iterations,
        active_config.n_subjects_per_iter,
        active_config.requested_fits_dir,
        active_config.requested_tfrecords_dir,
        active_config.train_records_index_loc
    )



if __name__ == '__main__':

    logging.basicConfig(
        filename='execute_{}.log'.format(time.time()),
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )

    parser = argparse.ArgumentParser(description='Execute active learning')
    parser.add_argument('--shard_config', dest='shard_config_loc', type=str,
                    help='Directory into which to place shard directory')
    parser.add_argument('--run_dir', dest='run_dir', type=str,
                    help='Path to csv catalog of Panoptes labels and fits_loc, for shards')
    parser.add_argument('--baseline', dest='baseline', type=bool, default=False,
                    help='Path to csv catalog of Panoptes labels and fits_loc, for shards')
    args = parser.parse_args()

    execute_active_learning(
        shard_config_loc=args.shard_config_loc,
        run_dir=args.run_dir,  # warning, may overwrite if not careful
        baseline=args.baseline
    )

