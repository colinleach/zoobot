"""Control what each active learning iteration will do. 
    Active learning parameters are expected (e.g. n subjects per iteration)
    Science parameters should not be included here!
    Oracle parameters (e.g. which zooniverse project) are okay
"""
import argparse
import os
import shutil
import logging
import json
import time
import json

import numpy as np

from shared_astro_utils import object_utils

from zoobot.active_learning import run_estimator_config, make_shards, acquisition_utils, oracles


class Instructions():

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
        Define those active learning parameters which are fixed between iterations (train/acquire cycles).
        Save those parameters to disk, so that EC2 instances can read them as required.
        Also, copy the database from shard_config_loc shard dir and update paths to images if needed.

        Args:
            shard_config_loc (str): path to json of shard metadata, e.g. location on disk, image size, etc. See `make_shards.py`
            save_dir (str): path to save run outputs e.g. trained models, new shards. NOT where instructions is saved!
            iterations (int): how many iterations to train the model (via train_callable)
            shards_per_iter (int): how many shards to find acquisition values for
            subjects_per_iter (int): how many subjects to acquire per training iteration
            initial_estimator_ckpt (str): path to checkpoint folder (datetime) of est. for initial iteration
        """
        # important to store all input args, to be able to save and restore from disk
        self.shard_config_loc = shard_config_loc  # useful to save, so we can restore Instructions from disk
        self.shards = make_shards.load_shard_config(shard_config_loc)
        self.save_dir = save_dir  # for database only, for now
        self.subjects_per_iter = subjects_per_iter
        self.shards_per_iter = shards_per_iter
        self.initial_estimator_ckpt = initial_estimator_ckpt
        self.n_samples = n_samples

        self.db_loc = os.path.join(self.save_dir, 'run_db.db')  

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # copy database
        if not os.path.exists(self.db_loc):  # may have already been copied TODO sloppy
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
        self.shards_per_iter = 1  # TODO should be 2
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
    logging.info(f'Loading instructions from {save_dir}')
    # necessary to reconstruct the instructions from disk in a new process before starting an iteration
    with open(os.path.join(save_dir, 'instructions.json'), 'r') as f:  # path must match save_instructions
        instructons_dict = json.load(f)
    return Instructions(**instructons_dict)


class TrainCallableFactory():

    def __init__(self, initial_size, final_size, warm_start, test):
        """
        Iteration (`iteration.py`) requires a callable that:
        - takes args which (may) that change each iteration (log_dir, train_records, etc)
        - trains a model

        TrainCallableFactory creates that callable via .get().
        TrainCallableFactory stores args which do *not* change every iteration (initial size, final_size, etc) 

        Note: highly coupled to iteration.py. Refactor inside?
        
        Args:
            initial_size (int): (fixed) size of galaxy images as serialized to tfrecord
            final_size (int): (desired) size of galaxy images after input pipeline
            warm_start (bool): If True, continue from the latest estimator in `log_dir`
            test (bool): If True, train on tiny images for a few epochs (i.e. run a functional test)
        """
        self.initial_size = initial_size
        self.final_size = final_size
        self.warm_start = warm_start
        self.test = test

    def get(self):
        """Using the fixed params stored in self, create a train callable (see class def).
        
        Returns:
            callable: callable expecting per-iteration args, training a model when called
        """
        def train_callable(log_dir, train_records, eval_records, learning_rate, epochs, schema, **kw_args):
            logging.info('Training model on: {}'.format(train_records))
            run_config = run_estimator_config.get_run_config(self, log_dir, train_records, eval_records, learning_rate, epochs, schema, **kw_args)
            if self.test: # overrides warm_start
                run_config.epochs = 2  # minimal training, for speed

            # Do NOT update eval_config: always eval on the same fixed shard
            return run_config.run_estimator()
        return train_callable

    def to_dict(self):
        return object_utils.object_to_dict(self)

    def save(self, save_dir):
        with open(os.path.join(save_dir, 'train_callable.json'), 'w') as f:  # path must match below
            json.dump(self.to_dict(), f)

def load_train_callable(save_dir):
    with open(os.path.join(save_dir, 'train_callable.json'), 'r') as f:  # path must match above
        data = json.load(f)
    return TrainCallableFactory(**data)


class AcquisitionCallableFactory():
    
    def __init__(self, baseline, expected_votes):
        """
        Analogous to TrainCallableFactory, but for the acquisition step following training.
        Iteration (`iteration.py`) requires a callable that:
        - takes args which (may) that change each iteration (here, subjects - the galaxies!)
        - calculates a priority for acquiring those subjects (see .get())

        AcquisitionCallableFactory creates that callable via .get().
        AcquisitionCallableFactory stores args which do *not* change every iteration (baseline, expected_votes) 

        Note: highly coupled to iteration.py. Refactor inside?

        Args:
            baseline (bool): if True, return random acquisition priorities
            expected_votes (int or iteratable): expected num. of responses to question per subject
        """
        self.baseline = baseline
        self.expected_votes = expected_votes

    def get(self):
        """Using the fixed params stored in self, create the acquisition callable.
        
        Returns:
            callable: acquisition callable (see __init__)
        """

        if self.baseline:
            logging.critical('Using mock acquisition function, baseline test mode!')
            return self.get_mock_acquisition_func()
        else:  # callable expecting samples np.ndarray, returning list
            logging.critical('Using mutual information acquisition function')
            return  lambda *args, **kwargs: np.mean(acquisition_utils.mutual_info_acquisition_func_multiq(*args, **kwargs), axis=-1)  # requires schema and retirement limit  

    def get_mock_acquisition_func(self):
        """Callable will return random subject priorities. Useful for baselines"""
        logging.critical('Retrieving MOCK random acquisition function')
        return lambda samples, *args, **kwargs: np.random.rand(len(samples))

    def to_dict(self):
        return object_utils.object_to_dict(self)

    def save(self, save_dir):
        with open(os.path.join(save_dir, 'acquisition_function.json'), 'w') as f:  # path must match load_instructions
            json.dump(self.to_dict(), f)

def load_acquisition_func(save_dir):
    with open(os.path.join(save_dir, 'acquisition_function.json'), 'r') as f:  # path must match above
        data = json.load(f)
    return AcquisitionCallableFactory(**data)


# this should be part of setup, so that db is modified to point to correct image folder
# TODO apply this to db after copying
# def get_relative_loc(loc, local_image_folder):
#     fname = os.path.basename(loc)
#     subdir = os.path.basename(os.path.dirname(loc))
#     return os.path.join(local_image_folder, subdir, fname)


def main(shard_config_loc, catalog_dir, instructions_dir, baseline, warm_start, test, panoptes):
    """
    Create a folder with all parameters that are fixed between active learning iterations.
    This is useful to read when an EC2 instance is spun up to run a new iteration.
    See `run_iteration.py` for use.
    
    The parameters are split into decoupled objects:
        - Instructions, general parameters defining how to run each active learning iteration
        - TrainCallableFactory, defining how to create train callables (for training an estimator)
        - AcquisitionCallableFactory, defining how to create acquisition callables (for prioritising subjects)

    Baseline, warm_start and test are args. All other parameters are hard-coded here, for now. 

    Args:
        shard_config_loc (str): 
        catalog_dir (str): dir holding catalogs to use. Needed to make oracle. See `prepare_catalogs.py`
        instructions_dir (str): directory to save the above parameters
        baseline (bool): if True, use random subject acquisition prioritisation
        warm_start (bool): if True, continue training the latest estimator from any log_dir provided to a train callable
        test (bool): if True, train on tiny images for a few iterations (i.e. run a functional test)
        panoptes (bool): if True, use Panoptes as oracle (upload subjects, download responses). Else, mock with historical responses.
    """
    # hardcoded defaults, for now
    subjects_per_iter = 512
    shards_per_iter = 2
    final_size = 128  # for both modes
    if baseline:
        n_samples = 2
    else:
        n_samples = 15

    expected_votes = 40  # SMOOTH MODE

    # decals
    # label_cols = [
    #     'smooth-or-featured_smooth',
    #     'smooth-or-featured_featured-or-disk',
    #     'has-spiral-arms_yes',
    #     'has-spiral-arms_no',
    #     'bar_strong',
    #     'bar_weak',
    #     'bar_no',
    #     'bulge-size_dominant',
    #     'bulge-size_large',
    #     'bulge-size_moderate',
    #     'bulge-size_small',
    #     'bulge-size_none'
    # ]

    # gz2 cols
    label_cols = [
        'smooth-or-featured_smooth',
        'smooth-or-featured_featured-or-disk',
        'has-spiral-arms_yes',
        'has-spiral-arms_no',
        'bar_yes',
        'bar_no',
        'bulge-size_dominant',
        'bulge-size_obvious',
        'bulge-size_just-noticeable',
        'bulge-size_no'
    ]

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
    train_callable_obj = TrainCallableFactory(
        initial_size=instructions.shards.size,
        final_size=final_size,
        warm_start=warm_start,
        test=test
    )
    train_callable_obj.save(instructions_dir)

    # parameters that only affect acquistion_func
    acquisition_func_obj = AcquisitionCallableFactory(
        baseline=baseline,
        expected_votes=expected_votes
    )
    acquisition_func_obj.save(instructions_dir)
    if panoptes: # use live Panoptes oracle
        oracle = oracles.Panoptes(
            catalog_loc=catalog_dir + '/unlabelled_catalog.csv',
            login_loc='zooniverse_login.json', 
            project_id='5733',
            workflow_ids=['6122', '10582'],
            last_id='160414882',  # TODO remove
            question='smooth'  # TODO sloppy!
        )
    else:  # use mock Panoptes oracle
        oracle_loc = catalog_dir + '/simulation_context/oracle.csv'
        assert os.path.isfile(oracle_loc)
        oracle = oracles.PanoptesMock(
            oracle_loc=oracle_loc,
            subjects_requested_loc=os.path.join(instructions_dir, 'subjects_requested.json'),
            label_cols=label_cols
        )
    oracle.save(instructions_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create instructions')
    parser.add_argument('--shard-config', dest='shard_config_loc', type=str,
                    help='Details of shards to use')
    parser.add_argument('--catalog-dir', dest='catalog_dir', type=str,
                    help='Details of shards to use')
                    #  TODO add catalog loc, to here or via shard config, to know which oracle to use
    parser.add_argument('--instructions-dir', dest='instructions_dir', type=str,
                    help='Directory to save instructions')
    parser.add_argument('--baseline', dest='baseline', action='store_true', default=False,
                    help='Use random subject selection only')
    parser.add_argument('--warm-start', dest='warm_start', action='store_true', default=False,
                    help='After each iteration, continue training the same model')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='Minimal training')
    parser.add_argument('--panoptes', dest='panoptes', action='store_true', default=False,
                    help='Use live uploads and responses')
    args = parser.parse_args()

    log_loc = 'create_instructions_{}.log'.format(time.time())

    logging.basicConfig(
        filename=log_loc,
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.warning('Baseline: {}'.format(args.baseline))

    main(args.shard_config_loc, args.catalog_dir, args.instructions_dir, args.baseline, args.warm_start, args.test, args.panoptes)

    # see run_simulation.sh for example use 
