import os
import shutil
import logging
import json

import numpy as np
import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params, setup
from zoobot.tests import TEST_EXAMPLE_DIR
from zoobot.tests import active_learning_test



class ShardConfig():
    # catalog, unlabelled shards, and single shard of labelled subjects
    # should be JSON serializable
    # at time of creation, many paths may not yet resolve - aimed at later run_dir
    def __init__(
        self,
        base_dir,  # to hold a new folder, named after the shard config 
        inital_size=64,
        final_size=28,
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
        # self.s3_fits_dir = '/users/mikewalmsley/aws/s3/galaxy-zoo/decals/fits_native'
        self.s3_fits_dir = None

        self.labelled_catalog_loc = os.path.join(self.shard_dir, 'labelled_catalog.csv')
        self.unlabelled_catalog_loc = os.path.join(self.shard_dir, 'unlabelled_catalog.csv')

        self.config_save_loc = os.path.join(self.shard_dir, 'shard_config.json')


    def prepare_shards(self, labelled_catalog, unlabelled_catalog):
        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir)  # always fresh
        os.mkdir(self.shard_dir)

        labelled_catalog['fits_loc'] = self.s3_fits_dir + labelled_catalog['fits_loc_relative']
        unlabelled_catalog['fits_loc'] = self.s3_fits_dir + unlabelled_catalog['fits_loc_relative']
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
        assert os.path.isdir(self.s3_fits_dir)
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

        self.max_iterations = 6
        self.n_subjects_per_iter = 1024


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


def snapshot_shards(volume_base_dir, catalog_loc):
    # in memory for now, but will be serialized for later/logs
    shard_config = ShardConfig(base_dir=volume_base_dir)  

    # in memory for now, but will be saved to csv
    catalog = pd.read_csv(catalog_loc)
    # >36 votes required, gives low count uncertainty
    catalog = catalog[catalog['smooth-or-featured_total-votes'] > 36]
    catalog['label'] = (catalog['smooth-or-featured_smooth_fraction'] > float(shard_config.label_split_value)).astype(int)  # 0 for featured
    catalog['id_str'] = catalog['subject_id'].astype(str) 

    # temporary hack for mocking panoptes
    labelled_catalog = catalog[:1024]  # for initial training data
    unlabelled_catalog = catalog[1024:1024*7]  # for new data
    del unlabelled_catalog['label']
    unlabelled_catalog.to_csv(os.path.join(TEST_EXAMPLE_DIR, 'panoptes.csv'))

    shard_config.prepare_shards(
        labelled_catalog,
        unlabelled_catalog)

    # must be able to end here, snapshot created and ready to go (hopefully)
    shard_config.write()


def execute_active_learning(shard_config_loc, run_dir, baseline=False):
    # on another machine, at another time...
    with open(shard_config_loc, 'r') as f:
        shard_config_dict = json.load(f)
    shard_config = ShardConfig(**shard_config_dict)
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
        active_config.requested_tfrecords_dir
    )


if __name__ == '__main__':

    # laptop_base = '/users/mikewalmsley/pretend_ec2_root'
    ec2_base = '/home/ec2-user'

    # laptop_catalog_loc = '/users/mikewalmsley/repos/zoobot/zoobot/tests/test_examples/panoptes_predictions.csv'
    ec2_catalog_loc = ec2_base + '/panoptes_predictions.csv'

    logging.basicConfig(
        filename='run_active_learning.log',
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG)

    snapshot_shards(
        volume_base_dir=ec2_base,
        catalog_loc=ec2_catalog_loc)

    # execute_active_learning(
    #     shard_config_loc='/users/mikewalmsley/pretend_ec2_root/shards_si64_sf28_l0.4/shard_config.json',
    #     run_dir='/users/mikewalmsley/pretend_ec2_root/run_baseline',
    #     baseline=True
    # )