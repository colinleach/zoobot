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
from collections import namedtuple

import numpy as np
import pandas as pd
import git

from shared_astro_utils import object_utils

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params, make_shards, analysis, iterations, acquisition_utils
from zoobot.tests import TEST_EXAMPLE_DIR


class Instructions():
    """
    Define and run active learning using a tensorflow estimator on pre-made shards
    (see make_shards.py for shard creation)
    """

    def __init__(
        self,
        shard_config_loc,
        save_dir,
        shards_per_iter,  # 4 mins per shard of 4096 images
        subjects_per_iter,
        initial_estimator_ckpt,
        n_samples,
        **overflow_args):  # when reloading, ignore any new properties not needed for __init__
        """
        Instructions for running each iteration, on existing shards

        Args:
            shard_config (ShardConfig): metadata of shards, e.g. location on disk, image size, etc.
            save_dir (str): path to save run outputs e.g. trained models, new shards
            iterations (int): how many iterations to train the model (via train_callable)
            shards_per_iter (int): how many shards to find acquisition values for
            subjects_per_iter (int): how many subjects to acquire per training iteration
            initial_estimator_ckpt (str): path to checkpoint folder (datetime) of est. for initial iteration
        """
        # important to store all input args, to be able to save and restore from disk
        self.shard_config_loc = shard_config_loc  # useful to save, so we can restore Instructions from disk
        self.shards = make_shards.load_shard_config(shard_config_loc)
        self.save_dir = save_dir
        self.subjects_per_iter = subjects_per_iter
        self.shards_per_iter = shards_per_iter
        self.initial_estimator_ckpt = initial_estimator_ckpt
        self.n_samples = n_samples

        self.db_loc = os.path.join(self.save_dir, 'run_db.db')  

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # copy database
        shutil.copyfile(self.shards.db_loc, self.db_loc)

    def ready(self):
        assert self.shards.ready()  # delegate
        if self.initial_estimator_ckpt is not None:
            assert os.path.isdir(self.initial_estimator_ckpt)
            assert os.path.exists(os.path.join(self.initial_estimator_ckpt, 'saved_model.pb'))
        assert os.path.isdir(self.save_dir)
        return True

    def use_test_mode(self):
        """Override for speed, to see if everything works as intended"""
        self.subjects_per_iter = 256
        self.shards_per_iter = 2
        self.shards.final_size = 32

    def to_dict(self):
        return object_utils.object_to_dict(self)

    def save(self, save_dir):
        # will lose shards, but save shard_config_loc so shards can be recovered
        with open(os.path.join(save_dir, 'instructions.json'), 'w') as f:  # path must match load_instructions
            data = self.to_dict()
            print(data)
            del data['shards']
            json.dump(data, f)

def load_instructions(save_dir):
    # necessary to reconstruct the instructions from disk in a new process before starting an iteration
    with open(os.path.join(save_dir, 'instructions.json'), 'r') as f:  # path must match save_instructions
        instructons_dict = json.load(f)
    return Instructions(**instructons_dict)


class TrainCallable():
    
    def __init__(self, initial_size, final_size, warm_start, eval_tfrecord_loc, test):
        self.initial_size = initial_size
        self.final_size = final_size
        self.warm_start = warm_start
        self.eval_tfrecord_loc = eval_tfrecord_loc
        self.test = test

    def get(self):
        def train_callable(log_dir, train_records, eval_records, learning_rate, epochs):
            logging.info('Training model on: {}'.format(train_records))
            run_config = default_estimator_params.get_run_config(self, log_dir, train_records, eval_records, learning_rate, epochs)
            if self.test: # overrides warm_start
                run_config.epochs = 2  # minimal training, for speed

            # Do NOT update eval_config: always eval on the same fixed shard
            return run_estimator.run_estimator(run_config)
        return train_callable

    def to_dict(self):
        return object_utils.object_to_dict(self)

    def save(self, save_dir):
        with open(os.path.join(save_dir, 'train_callable.json'), 'w') as f:  # path must match below
            json.dump(self.to_dict(), f)

def load_train_callable(save_dir):
    with open(os.path.join(save_dir, 'train_callable.json'), 'r') as f:  # path must match above
        data = json.load(f)
    return TrainCallable(**data)


class AcquisitionFunction():
    
    def __init__(self, baseline, expected_votes):
        self.baseline = baseline
        self.expected_votes = expected_votes

    def get(self):
        if self.baseline:
            logging.critical('Using mock acquisition function, baseline test mode!')
            return self.get_mock_acquisition_func()
        else:  # callable expecting samples np.ndarray, returning list
            logging.critical('Using mutual information acquisition function')
            return lambda x: acquisition_utils.mutual_info_acquisition_func(x, self.expected_votes)  

    def get_mock_acquisition_func(self):
        logging.critical('Retrieving MOCK random acquisition function')
        return lambda samples: [np.random.rand() for n in range(len(samples))]

    def to_dict(self):
        return object_utils.object_to_dict(self)

    def save(self, save_dir):
        with open(os.path.join(save_dir, 'acquisition_function.json'), 'w') as f:  # path must match load_instructions
            json.dump(self.to_dict(), f)

def load_acquisition_func(save_dir):
    with open(os.path.join(save_dir, 'acquisition_function.json'), 'r') as f:  # path must match above
        data = json.load(f)
    return AcquisitionFunction(**data)


# this should be part of setup, so that db is modified to point to correct image folder
# TODO apply this to db after copying
def get_relative_loc(loc, local_image_folder):
    fname = os.path.basename(loc)
    subdir = os.path.basename(os.path.dirname(loc))
    return os.path.join(local_image_folder, subdir, fname)


def main(shard_config_loc, instructions_dir, baseline, warm_start, test):

    # hardcoded defaults, for now
    subjects_per_iter = 128
    shards_per_iter = 4
    final_size = 128  # for both modes
    if baseline:
        n_samples = 2
    else:
        n_samples = 15

    expected_votes = 40  # SMOOTH MODE

    # record instructions
    instructions = Instructions(
        shard_config_loc, 
        instructions_dir,
        subjects_per_iter=subjects_per_iter,
        shards_per_iter=shards_per_iter,
        initial_estimator_ckpt=None,
        n_samples=n_samples
    )
    instructions.save(instructions_dir)

    # parameters that only affect train_callable
    train_callable_obj = TrainCallable(
        initial_size=instructions.shards.size,
        final_size=final_size,
        warm_start=warm_start,
        # TODO remove?
        eval_tfrecord_loc=instructions.shards.eval_tfrecord_locs(),
        test=test
    )
    train_callable_obj.save(instructions_dir)

    # parameters that only affect acquistion_func
    acquisition_func_obj = AcquisitionFunction(
        baseline=baseline,
        expected_votes=expected_votes
    )
    acquisition_func_obj.save(instructions_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Execute active learning')
    parser.add_argument('--shard-config', dest='shard_config_loc', type=str,
                    help='Details of shards to use')
    parser.add_argument('--instructions-dir', dest='instructions_dir', type=str,
                    help='Directory to save instructions')
    parser.add_argument('--baseline', dest='baseline', action='store_true', default=False,
                    help='Use random subject selection only')
    parser.add_argument('--warm-start', dest='warm_start', action='store_true', default=False,
                    help='After each iteration, continue training the same model')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='Minimal training')
    args = parser.parse_args()

    log_loc = 'create_instructions_{}.log'.format(time.time())

    logging.basicConfig(
        filename=log_loc,
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    main(args.shard_config_loc, args.instructions_dir, args.baseline, args.warm_start, args.test)

