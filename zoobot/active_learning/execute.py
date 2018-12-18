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
import git

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params, make_shards, analysis, mock_panoptes
from zoobot.tests import TEST_EXAMPLE_DIR
from zoobot.tests.active_learning import active_learning_test


class ActiveConfig():
    """
    Define and run active learning using a tensorflow estimator on pre-made shards
    (see make_shards.py for shard creation)
    """


    def __init__(
        self,
        shard_config,
        run_dir,
        iterations, 
        shards_per_iter,  # 4 mins per shard of 4096 images
        subjects_per_iter,
        warm_start=False):
        """
        Controller to define and run active learning on pre-made shards

        To use:
        active_config.prepare_run_folders()
        assert active_config.ready()
        active_config.run(
            train_callable,
            get_acquisition_func
        )
        For the form of train_callable and get_acquisition func, see active_config.run
        
        Args:
            shard_config (ShardConfig): metadata of shards, e.g. location on disk, image size, etc.
            run_dir (str): path to save run outputs e.g. trained models, new shards
            iterations (int): how many iterations to train the model (via train_callable)
            shards_per_iter (int): how many shards to find acquisition values for
            subjects_per_iter (int): how many subjects to acquire per training iteration
        """
        self.shards = shard_config
        self.run_dir = run_dir

        self.iterations = iterations  
        self.subjects_per_iter = subjects_per_iter
        self.shards_per_iter = shards_per_iter

        self.warm_start = warm_start

        self.db_loc = os.path.join(self.run_dir, 'run_db.db')  
        self.estimator_dir = os.path.join(self.run_dir, 'estimator')
    
        # will download/copy fits of top acquisitions into here
        self.requested_fits_dir = os.path.join(self.run_dir, 'requested_fits')
        # and then write them into tfrecords here
        self.requested_tfrecords_dir = os.path.join(self.run_dir, 'requested_tfrecords')

        # state for train records is entirely on disk, to allow for warm starts
        self.train_records_index_loc = os.path.join(self.run_dir, 'requested_tfrecords_index.json')
        if not self.warm_start or not os.path.isfile(self.train_records_index_loc):
            self.write_train_records_index([self.shards.train_tfrecord_loc])  # will append new shards


    def prepare_run_folders(self):
        """
        Create the folders needed to run active learning. 
        If self.warm_start is False, if any already exist, then wipe them.
        """
        # order is important due to rmtree
        directories = [self.run_dir, self.estimator_dir, self.requested_fits_dir, self.requested_tfrecords_dir]

        # if warm start, check all directories exist and, if not, make them.
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

        # copy initial shard db to run directory, to modify, if not already copied (warm start)
        if not os.path.isfile(self.db_loc):
            shutil.copyfile(self.shards.db_loc, self.db_loc)  


    def ready(self):
        assert self.shards.ready()
        assert os.path.isdir(self.estimator_dir)
        assert os.path.isdir(self.run_dir)
        assert os.path.isdir(self.requested_fits_dir)
        assert os.path.isdir(self.requested_tfrecords_dir)
        return True


    def run(self, train_callable, get_acquisition_func):
        """Main active learning training loop. 
        
        Learn with train_callable
        Calculate acquisition functions for each subject in the shards
        Load .fits of top subjects and save to a new shard
        Repeat for self.iterations
        After each iteration, copy the model history to new directory and start again
        Designed to work with tensorflow estimators
        
        Args:
            train_callable (func): train a tf model. Arg: list of tfrecord locations
            get_acquisition_func (func): Make callable for acq. func. Arg: trained & loaded tf model
        """
        db = sqlite3.connect(self.db_loc)
        shard_locs = itertools.cycle(active_learning.get_all_shard_locs(db))  # cycle through shards

        if self.warm_start:
            shutil.rmtree(self.estimator_dir)  # do not restore from the estimator dir itself, only from complete iterations
            latest_model_dir = self.get_most_recent_model_loc()
            shutil.copytree(latest_model_dir, self.estimator_dir)  # put model from latest complete iteration in estimator dir

        iteration = 0
        while iteration < self.iterations:
            # train as usual, with saved_model being placed in estimator_dir
            logging.info('Training iteration {}'.format(iteration))
        
            # callable should expect list of tfrecord files to train on
            train_callable(self.get_train_records())  # could be docker container to run, save model

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
                logging.info('Using shard_loc {}, iteration {}, max {}'.format(shard_loc, iteration, self.shards_per_iter))
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
            self.add_train_record(new_train_tfrecord)

            if not self.warm_start:
                # copy estimator directory to run_dir, and make a new empty estimator_dir
                # otherwise, estimator will reload from the end of the previous iteration
                shutil.move(self.estimator_dir, os.path.join(self.run_dir, 'iteration_{}'.format(iteration)))
                os.mkdir(self.estimator_dir)

            iteration += 1


    def get_train_records(self):
        logging.info('Attempting to load {}'.format(self.train_records_index_loc))
        with open(self.train_records_index_loc, 'r') as f:  # must exist, see __init__
            train_records = json.load(f)  # restore from disk all previous train records
        logging.info('Loaded train records: {}'.format(train_records))
        assert isinstance(train_records, list)
        return train_records

    def add_train_record(self, new_record_loc):
        # must always be kept in sync
        current_records = self.get_train_records()
        current_records.append(new_record_loc)
        self.write_train_records_index(current_records)


    def write_train_records_index(self, train_records):
        with open(self.train_records_index_loc, 'w') as f:
            json.dump(train_records, f)


    def get_most_recent_model_loc(self):
        # latest estimator dir will be iteration_n for max(n)
        # anything in estimator_dir itself is not restored
        # warning - strongly coupled to self.run() last paragraphs
        iteration = 0
        latest_estimator_dir = os.path.join(self.run_dir, 'iteration_0')
        while True:
            dir_to_test = os.path.join(self.run_dir, 'iteration_{}'.format(iteration))
            if not os.path.isdir(dir_to_test):
                break
            else:
                latest_estimator_dir = dir_to_test
                iteration += 1
        # get the latest checkpoint in that estimator dir
        logging.info('latest estimator dir is {}'.format(latest_estimator_dir))
        latest_model_loc = active_learning.get_latest_checkpoint_dir(latest_estimator_dir)
        logging.info('Found latest model: {}'.format(latest_model_loc))
        return latest_model_loc



def execute_active_learning(shard_config_loc, run_dir, baseline=False, test=False, warm_start=False):
    """
    Train a model using active learning, on the data (shards) described in shard_config_loc
    Run parameters (except shards) are defined here and in default_estimator_params.get_run_config

    Args:
        shard_config_loc ([type]): path to shard config (json) describing existing shards to use
        run_dir (str): output directory to save model and new shards
        baseline (bool, optional): Defaults to False. If True, use random selection for acquisition.
        warm_start(bool, optional): Defaults to False. If True, preserve model between iterations
    
    Returns:
        None
    """
    if test:  # do a brief run only
        iterations = 3
        subjects_per_iter = 28
        shards_per_iter = 1
    else:
        iterations = 4  # 1.5h per iteration
        subjects_per_iter = 512
        shards_per_iter = 3

    shard_config = make_shards.load_shard_config(shard_config_loc)
    # instructions for the run (except model)
    active_config = ActiveConfig(
        shard_config, 
        run_dir,
        iterations=iterations, 
        subjects_per_iter=subjects_per_iter,
        shards_per_iter=shards_per_iter,
        warm_start=warm_start
    )  
    active_config.prepare_run_folders()

    run_config = default_estimator_params.get_run_config(active_config)  # instructions for model

    if active_config.warm_start:
        run_config.epochs = 75  # for retraining
    if test: # overrides warm_start
        run_config.epochs = 5  # minimal training, for speed

    def train_callable(train_records):
        run_config.train_config.tfrecord_loc = train_records
        # Do NOT update eval_config: always eval on the same fixed shard
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
        get_acquisition_func = lambda predictor: make_predictions.get_acquisition_func(predictor, n_samples=20)  # predictor should be directory of saved_model.pb

    unlabelled_catalog = pd.read_csv(
        active_config.shards.unlabelled_catalog_loc, 
        dtype={'id_str': str, 'label': int}
    )

    active_config.run(
        train_callable,
        get_acquisition_func
    )

    analysis.show_subjects_by_iteration(active_config.train_records_index_loc, 15, 128, 3, os.path.join(active_config.run_dir, 'subject_history.png'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Execute active learning')
    parser.add_argument('--shard_config', dest='shard_config_loc', type=str,
                    help='Details of shards to use')
    parser.add_argument('--run_dir', dest='run_dir', type=str,
                    help='Path to save run outputs: models, new shards, log')
    parser.add_argument('--baseline', dest='baseline', type=bool, default=False,
                    help='Use random subject selection only')
    parser.add_argument('--test', dest='test', type=bool, default=False,
                    help='Only do a minimal run to verify that everything works')
    parser.add_argument('--warm-start', dest='warm_start', type=bool, default=False,
                    help='After each iteration, continue training the same model')
    args = parser.parse_args()

    log_loc = 'execute_{}.log'.format(time.time())

    logging.basicConfig(
        filename=log_loc,
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )

    execute_active_learning(
        shard_config_loc=args.shard_config_loc,
        run_dir=args.run_dir,  # warning, may overwrite if not careful
        baseline=args.baseline,
        test=args.test,
        warm_start=args.warm_start
    )

    # finally, tidy up by moving the log into the run directory
    # could not be create here because run directory did not exist at start of script
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    shutil.move(log_loc, os.path.join(args.run_dir, '{}.log'.format(sha)))
