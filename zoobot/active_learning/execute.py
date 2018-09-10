import argparse
import os
import shutil
import logging
import json
import time
import sqlite3
import json
import itertools

import numpy as np
import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params, make_shards, analysis, mock_panoptes
from zoobot.tests import TEST_EXAMPLE_DIR
from zoobot.tests.active_learning import active_learning_test


class ActiveConfig():

    def __init__(
        self,
        shard_config,
        run_dir,
        iterations=6, 
        subjects_per_iter=1024,
        shards_per_iter=4,
        warm_start=False,  # warning
        restart_each_iter=True):  # warning

        self.shards = shard_config
        self.run_dir = run_dir

        self.iterations = iterations
        self.subjects_per_iter = subjects_per_iter
        self.shards_per_iter = shards_per_iter

        self.warm_start = warm_start
        self.restart_each_iter = restart_each_iter

        self.db_loc = os.path.join(self.run_dir, 'run_db.db')  
        self.estimator_dir = os.path.join(self.run_dir, 'estimator')
    
        # will download/copy fits of top acquisitions into here
        self.requested_fits_dir = os.path.join(self.run_dir, 'requested_fits')
        # and then write them into tfrecords here
        self.requested_tfrecords_dir = os.path.join(self.run_dir, 'requested_tfrecords')
        self.train_records_index_loc = os.path.join(self.run_dir, 'requested_tfrecords_index.json')



    def prepare_run_folders(self):
        # new predictors (apart from initial disk load) for now

        # order is important due to rmtree
        directories = [self.run_dir, self.estimator_dir, self.requested_fits_dir, self.requested_tfrecords_dir]

        # if warm start, check all exist and, if not, make.
        if self.warm_start:
            for directory in directories:
                if not os.path.isdir(directory):
                    os.mkdir(directory)

        # if not warm start, delete root and remake all
        if not self.warm_start:
            for directory in directories:
                if os.path.isdir(directory):
                    shutil.rmtree(directory)
            for directory in directories:
                os.mkdir(directory)

        if not os.path.isfile(self.db_loc):
            shutil.copyfile(self.shards.db_loc, self.db_loc)  # copy initial shard db to here, to modify


    def ready(self):
        assert self.shards.ready()
        # TODO more validation checks for the run
        assert os.path.isdir(self.estimator_dir)
        assert os.path.isdir(self.run_dir)
        assert os.path.isdir(self.requested_fits_dir)
        assert os.path.isdir(self.requested_tfrecords_dir)

        return True


    def run(self, catalog, train_callable, get_acquisition_func):
        assert 'label' not in catalog.columns.values

        db = sqlite3.connect(self.db_loc)
        shard_locs = itertools.cycle(active_learning.get_all_shard_locs(db))  # cycle through shards

        train_records = [self.shards.train_tfrecord_loc]  # will append new train records (save to db?)
        iteration = 0
        while iteration < self.iterations:
            # train as usual, with saved_model being placed in estimator_dir
            logging.info('Training iteration {}'.format(iteration))
        
            # callable should expect list of tfrecord files to train on
            train_callable(train_records)  # could be docker container to run, save model

            # make predictions and save to db, could be docker container
            predictor_loc = active_learning.get_latest_checkpoint_dir(self.estimator_dir)
            logging.info('Loading model from {}'.format(predictor_loc))
            predictor = make_predictions.load_predictor(predictor_loc)
            # inner acq. func is derived from make predictions acq func but with predictor and n_samples set
            acquisition_func = get_acquisition_func(predictor)

            logging.info('Making and recording predictions')
            shards_used = 0
            top_acquisition_ids = []
            while shards_used < self.shards_per_iter:
                shard_loc = next(shard_locs)
                logging.info('Using shard_loc {}, iteration {}'.format(shard_loc, iteration))
                active_learning.record_acquisitions_on_tfrecord(db, shard_loc, self.shards.initial_size, self.shards.channels, acquisition_func)
                shards_used += 1

            top_acquisition_ids = active_learning.get_top_acquisitions(db, self.subjects_per_iter, shard_loc=None)

            labels = mock_panoptes.get_labels(top_acquisition_ids)

            active_learning.add_labels_to_db(top_acquisition_ids, labels, db)

            new_train_tfrecord = os.path.join(self.requested_tfrecords_dir, 'acquired_shard_{}.tfrecord'.format(iteration))
            logging.info('Saving top acquisition subjects to {}, labels: {}...'.format(new_train_tfrecord, labels[:5]))
            # TODO download the top acquisitions from S3 if not on local system, for EC2
            # Can do this when switching to production, not necessary to demonstrate the system
            # fits_loc_s3 column?
            active_learning.add_labelled_subjects_to_tfrecord(db, top_acquisition_ids, new_train_tfrecord, self.shards.initial_size)
            train_records.append(new_train_tfrecord)

            with open(self.train_records_index_loc, 'w') as f:
                json.dump(train_records, f)

            if self.restart_each_iter:
                # copy estimator directory to run_dir, and make a new empty estimator_dir
                shutil.move(self.estimator_dir, os.path.join(self.run_dir, 'iteration_{}'.format(iteration)))
                os.mkdir(self.estimator_dir)

            iteration += 1




def execute_active_learning(shard_config_loc, run_dir, baseline=False):
    shard_config = make_shards.load_shard_config(shard_config_loc)
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

    active_config.run(
        unlabelled_catalog, 
        train_callable,
        get_acquisition_func
    )

    analysis.show_subjects_by_iteration(active_config.train_records_index_loc, 15, 128, 3, os.path.join(active_config.run_dir, 'subject_history.png'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Execute active learning')
    parser.add_argument('--shard_config', dest='shard_config_loc', type=str,
                    help='Directory into which to place shard directory')
    parser.add_argument('--run_dir', dest='run_dir', type=str,
                    help='Path to csv catalog of Panoptes labels and fits_loc, for shards')
    parser.add_argument('--baseline', dest='baseline', type=bool, default=False,
                    help='Path to csv catalog of Panoptes labels and fits_loc, for shards')
    args = parser.parse_args()

    log_loc = '/home/ubuntu/execute_{}.log'.format(time.time())

    logging.basicConfig(
        filename=log_loc,
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )

    logging.warning('Saving logs to: ' + log_loc)

    execute_active_learning(
        shard_config_loc=args.shard_config_loc,
        run_dir=args.run_dir,  # warning, may overwrite if not careful
        baseline=args.baseline
    )

