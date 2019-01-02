import argparse
import os
import shutil
import logging
import json
import time
import sqlite3
import json
import subprocess
import itertools

import numpy as np
import pandas as pd
import git

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params, make_shards, analysis, iterations, acquisition_utils
from zoobot.tests import TEST_EXAMPLE_DIR

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
        initial_estimator_ckpt,
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
            initial_estimator_ckpt (str): path to checkpoint folder (datetime) of est. for initial iteration
            warm_start (bool): if True, continue training the same estimator between iterations. Else, start from scratch.
        """
        self.shards = shard_config
        self.run_dir = run_dir

        self.iterations = iterations  
        self.subjects_per_iter = subjects_per_iter
        self.shards_per_iter = shards_per_iter

        self.initial_estimator_ckpt = initial_estimator_ckpt
        self.warm_start = warm_start

        self.db_loc = os.path.join(self.run_dir, 'run_db.db')  
    
        # will download/copy fits of top acquisitions into here
        self.requested_fits_dir = os.path.join(self.run_dir, 'requested_fits')
        # and then write them into tfrecords here
        self.requested_tfrecords_dir = os.path.join(self.run_dir, 'requested_tfrecords')

        # state for train records is entirely on disk, to allow for warm starts
        self.train_records_index_loc = os.path.join(self.run_dir, 'requested_tfrecords_index.json')


    def prepare_run_folders(self):
        """
        Create the folders needed to run active learning. 
        Copy the shard database, to be modified by the run
        Wipes any existing folders in run_dir
        """
        assert os.path.exists(self.run_dir)
        shutil.rmtree(self.run_dir)

        directories = [self.run_dir, self.requested_fits_dir, self.requested_tfrecords_dir]
        for directory in directories:
            os.mkdir(directory)

        shutil.copyfile(self.shards.db_loc, self.db_loc)


    def ready(self):
        assert self.shards.ready()  # delegate
        assert os.path.isfile(self.train_records_index_loc)
        if self.initial_estimator_ckpt is not None:
            assert os.path.isdir(self.initial_estimator_ckpt)
            assert os.path.exists(os.path.join(self.initial_estimator_ckpt, 'saved_model.pb'))
        assert os.path.isdir(self.run_dir)
        assert os.path.isdir(self.requested_fits_dir)
        assert os.path.isdir(self.requested_tfrecords_dir)
        assert os.path.isfile(self.train_records_index_loc)
        return True


    def run(self, train_callable, acquisition_func):
        """Main active learning training loop. 
        
        Learn with train_callable
        Calculate acquisition functions for each subject in the shards
        Load .fits of top subjects and save to a new shard
        Repeat for self.iterations
        After each iteration, copy the model history to new directory and start again
        Designed to work with tensorflow estimators
        
        Args:
            train_callable (func): train a tf model. Arg: list of tfrecord locations
            acquisition_func (func): expecting samples of shape [n_subject, n_sample]
        """
        assert self.ready()
        db = sqlite3.connect(self.db_loc)
        shards_iterable = itertools.cycle(active_learning.get_all_shard_locs(db))  # cycle through shards

        iteration_n = 0
        initial_estimator_ckpt = self.initial_estimator_ckpt  # for first iteration, the first model is the one passed to ActiveConfig
        initial_db_loc = self.db_loc
        initial_train_tfrecords=self.shards.initial_train_tfrecords
        
        while iteration_n < self.iterations:

            prediction_shards = [next(shards_iterable) for n in range(self.shards_per_iter)]

            iteration = iterations.Iteration(
                run_dir=self.run_dir, 
                iteration_n=iteration_n, 
                prediction_shards=prediction_shards,
                initial_db_loc=initial_db_loc,
                initial_train_tfrecords=initial_train_tfrecords,
                train_callable=train_callable,
                acquisition_func=acquisition_func,
                n_samples=20,
                n_subjects_to_acquire=self.subjects_per_iter,
                initial_size=self.shards.initial_size,
                initial_estimator_ckpt=initial_estimator_ckpt)

            # train as usual, with saved_model being placed in estimator_dir
            logging.info('Training iteration {}'.format(iteration_n))
            iteration.run()

            iteration_n += 1
            initial_db_loc = iteration.db_loc
            initial_train_tfrecords = iteration.get_train_records()  # includes newly acquired shard
            initial_estimator_ckpt = active_learning.get_latest_checkpoint_dir(iteration.estimators_dir)


    def get_most_recent_iteration_loc(self):
        # latest estimator dir will be iteration_n for max(n)
        # anything in estimator_dir itself is not restored
        # warning - strongly coupled to self.run() last paragraphs
        iteration_n = 0
        latest_estimator_dir = os.path.join(self.run_dir, 'iteration_0')
        while True:
            dir_to_test = os.path.join(self.run_dir, 'iteration_{}'.format(iteration_n))
            if not os.path.isdir(dir_to_test):
                break
            else:
                latest_estimator_dir = dir_to_test
                iteration_n += 1
        logging.info('latest estimator dir is {}'.format(latest_estimator_dir))
        return latest_estimator_dir


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
        iterations = 2
        subjects_per_iter = 28
        shards_per_iter = 1
    else:
        iterations = 5  # 1.5h per iteration
        subjects_per_iter = 1024
        shards_per_iter = 3

    shard_config = make_shards.load_shard_config(shard_config_loc)
    # instructions for the run (except model)
    active_config = ActiveConfig(
        shard_config, 
        run_dir,
        iterations=iterations, 
        subjects_per_iter=subjects_per_iter,
        shards_per_iter=shards_per_iter,
        initial_estimator_ckpt=None,
        warm_start=warm_start
    )
    active_config.prepare_run_folders()

    # WARNING run_config is not actually part of active_config, only interacts via callables
    run_config = default_estimator_params.get_run_config(active_config)  # instructions for model

    if active_config.warm_start:
        run_config.epochs = 150  # for retraining
    if test: # overrides warm_start
        run_config.epochs = 5  # minimal training, for speed

    def train_callable(log_dir, train_records):
        run_config.log_dir = log_dir
        run_config.train_config.tfrecord_loc = train_records
        # Do NOT update eval_config: always eval on the same fixed shard
        return run_estimator.run_estimator(run_config)

    if baseline:
        def mock_acquisition_func(samples):
            return samples.mean(axis=1)
        logging.warning('Using mock acquisition function, baseline test mode')
        acquisition_func = mock_acquisition_func
    else:  # callable expecting predictor, returning a callable expecting matrix list
        acquisition_func = acquisition_utils.mutual_info_acquisition_func  # predictor should be directory of saved_model.pb

    active_config.run(
        train_callable,
        acquisition_func
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
