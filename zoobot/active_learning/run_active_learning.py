import os
import shutil
import logging
import json

import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params, setup
from zoobot.tests import TEST_EXAMPLE_DIR



class ShardConfig():
    # catalog, unlabelled shards, and single shard of labelled subjects
    # should be JSON serializable
    # at time of creation, many paths may not yet resolve - aimed at later run_dir
    def __init__(
        self, 
        base_dir,  # to hold a new folder, named after the shard config 
        inital_size=64, 
        final_size=28, 
        shard_size=25, 
        label_split_value='0.4',
        **overflow_args
        ):

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

        self.train_tfrecord_loc = os.path.join(self.shard_dir, self.run_name, 'initial_train.tfrecord')
        self.eval_tfrecord_loc = os.path.join(self.shard_dir, self.run_name, 'eval.tfrecord')

        self.labelled_catalog_loc = os.path.join(self.shard_dir, 'labelled_catalog.csv')
        self.unlabelled_catalog_loc = os.path.join(self.shard_dir, 'unlabelled_catalog.csv')

        # TODO move to shared utilities
        def asdict(self):
            excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
            return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])

        
        def to_disk(self, disk_loc):
            json.dump(self.as_dict, disk_loc)


    def prepare_shards(self, labelled_catalog, unlabelled_catalog):

        assert os.path.exists(self.shard_dir)
        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir)  # always fresh
        os.mkdir(self.shard_dir)

        labelled_catalog.to_csv(self.labelled_catalog_loc)
        unlabelled_catalog.to_csv(self.unlabelled_catalog_loc)

        setup.make_initial_training_tfrecord(
            labelled_catalog, 
            self.train_tfrecord_loc, 
            self.initial_size)
        # TODO also write new eval tfrecord based on known catalog. Important!

        setup.make_database_and_shards(
            unlabelled_catalog, 
            self.db_loc, 
            self.initial_size, 
            self.shard_dir, 
            self.shard_size)



class ActiveConfig():

    def __init__(self, shard_config, run_dir):
        self.shards = shard_config
        self.run_dir = run_dir
        self.db_loc = os.path.join(self.run_dir, 'run_db.db')  # will copy static to here
        self.predictor_dir = os.path.join(self.run_dir, 'estimator')  # TODO rename from predictor_dir
        self.labelled_shard_dir = os.path.join(self.run_dir, 'labelled_shards')

        # shard config not yet used, but surely needed

    def prepare_run_folders(self):
        os.mkdir(self.run_dir)
        # new predictors (apart from initial disk load) for now
        if os.path.exists(self.predictor_dir):
            shutil.rmtree(self.predictor_dir)
        os.mkdir(self.predictor_dir)


    def ready_to_run(self):
        assert os.path.exists(self.shards.train_tfrecord_loc)
        assert os.path.exists(self.shards.eval_tfrecord_loc)
        return True
        # TODO more validation checks


def benchmark_active_learning(volume_base_dir, catalog_loc):
    # in memory for now, but will be serialized for later/logs
    shard_config = ShardConfig(base_dir=volume_base_dir)  

    # in memory for now, but will be saved to csv
    catalog = pd.read_csv(catalog_loc)
    # >36 votes required, gives low count uncertainty
    catalog = catalog[catalog['smooth-or-featured_total-votes'] > 36]
    catalog['label'] = (catalog['smooth-or-featured_smooth_fraction'] > float(shard_config.label_split_value)).astype(int)  # 0 for featured
    catalog['id_str'] = catalog['subject_id'].astype(str) 

    labelled_catalog = catalog[:50]  # for initial training data
    unlabelled_catalog = catalog[50:100]  # for new data
    unlabelled_catalog.to_csv(os.path.join(TEST_EXAMPLE_DIR, 'panoptes.csv'))

    shard_config.prepare_shards(
        labelled_catalog,
        unlabelled_catalog)
    # must be able to end here, snapshot created and ready to go (hopefully)


def execute_active_learning(shard_config_loc, run_dir):
    # on another machine, at another time...
    shard_config_dict = json.load(shard_config_loc)
    shard_config = ShardConfig(**shard_config_dict)
    active_config = ActiveConfig(shard_config, run_dir)
    active_config.prepare_run_folders()
    assert active_config.ready_to_run()
    # define the estimator - load settings (rename 'setup' to 'settings'?)
    run_config = default_estimator_params.get_run_config(active_config)
    train_callable = lambda: run_estimator.run_estimator(run_config)
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
        active_config.predictor_dir, 
        active_config.shards.train_tfrecord_loc, 
        train_callable, 
        get_acquisition_func
    )


if __name__ == '__main__':

    logging.basicConfig(
        filename='run_active_learning.log',
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG)

    benchmark_active_learning(
        volume_base_dir='/users/mikewalmsley/pretend_ec2_root',
        catalog_loc='/users/mikewalmsley/data/galaxy_zoo/decals/panoptes_mock_predictions.csv')
