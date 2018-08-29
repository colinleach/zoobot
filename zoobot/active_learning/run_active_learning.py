import os
import shutil
import logging
import json

import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params, setup
from zoobot.tests import TEST_EXAMPLE_DIR


class ActiveLearningConfig():
    # should be JSON serializable
    # at time of creation, many paths may not yet resolve - aimed at later run_data_dir
    def __init__(
        self, 
        base_dir, 
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

        self.run_name = 'active_si{}_sf{}_l{}'.format(
            self.initial_size, 
            self.final_size, 
            self.label_split_value
        )

        self.base_dir = base_dir
        self.static_data_dir = os.path.join(self.base_dir, 'static_data')
        self.run_data_dir = os.path.join(self.base_dir, 'run')

        self.static_db_loc = os.path.join(self.run_data_dir, 'static_shard_db.db')  # assumed
        self.db_loc = os.path.join(self.run_data_dir, 'run_db.db')  # will copy static to here
        # copy operation now?
        self.predictor_dir = os.path.join(self.run_data_dir, 'estimator')  # TODO rename from predictor_dir
        self.labelled_shard_dir = os.path.join(self.run_data_dir, 'labelled_shards')
        self.unlabelled_shard_dir = os.path.join(self.static_data_dir, 'unlabelled_shards')

        # need to rename labelled shard v unlabelled shard, not a good mapping
        self.train_tfrecord_loc = os.path.join(self.static_data_dir, self.run_name, 'initial_train.tfrecord')
        self.eval_tfrecord_loc = os.path.join(self.static_data_dir, self.run_name, 'eval.tfrecord')

        self.labelled_catalog_loc = os.path.join(self.static_data_dir, 'labelled_catalog.csv')
        self.unlabelled_catalog_loc = os.path.join(self.static_data_dir, 'unlabelled_catalog.csv')

        # TODO move to shared utilities
        def asdict(self):
            excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
            return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])

        
        def to_disk(self, disk_loc):
            json.dump(self.as_dict, disk_loc)


    def prepare_snapshot(self, labelled_catalog, unlabelled_catalog):

        assert os.path.exists(self.base_dir)
        if os.path.isdir(self.static_data_dir):
            shutil.rmtree(self.static_data_dir)  # always fresh
        os.mkdir(self.static_data_dir)
        os.mkdir(self.unlabelled_shard_dir)

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
            self.unlabelled_shard_dir, 
            self.shard_size)

    
    def prepare_run_folders(self):

        os.mkdir(self.run_data_dir)
        os.mkdir(self.labelled_shard_dir)
        # new predictors (apart from initial disk load) for now
        if os.path.exists(self.predictor_dir):
            shutil.rmtree(self.predictor_dir)
        os.mkdir(self.predictor_dir)


    def ready_to_run(self):
        assert os.path.exists(self.train_tfrecord_loc)
        assert os.path.exists(self.eval_tfrecord_loc)
        return True
        # TODO more validation checks


def benchmark_active_learning(volume_base_dir, catalog_loc):
    # in memory for now, but will be serialized for later/logs
    active_config = ActiveLearningConfig(base_dir=volume_base_dir)  

    # in memory for now, but will be saved to csv
    catalog = pd.read_csv(catalog_loc)
    # >36 votes required, gives low count uncertainty
    catalog = catalog[catalog['smooth-or-featured_total-votes'] > 36]
    catalog['label'] = (catalog['smooth-or-featured_smooth_fraction'] > float(active_config.label_split_value)).astype(int)  # 0 for featured
    catalog['id_str'] = catalog['subject_id'].astype(str) 

    labelled_catalog = catalog[:50]  # for initial training data
    unlabelled_catalog = catalog[50:100]  # for new data
    unlabelled_catalog.to_csv(os.path.join(TEST_EXAMPLE_DIR, 'panoptes.csv'))

    active_config.prepare_snapshot(
        labelled_catalog,
        unlabelled_catalog)
    # must be able to end here, snapshot created and ready to go (hopefully)


def execute_active_learning(active_config_loc):
    # on another machine, at another time...
    active_config_dict = json.load(active_config_loc)
    active_config = ActiveLearningConfig(**active_config_dict)
    active_config.prepare_run_folders()
    assert active_config.ready_to_run()
    # define the estimator - load settings (rename 'setup' to 'settings'?)
    run_config = default_estimator_params.get_run_config(active_config)
    train_callable = lambda: run_estimator.run_estimator(run_config)
    get_acquisition_func = lambda predictor: make_predictions.get_acquisition_func(predictor, n_samples=20)
    unlabelled_catalog = pd.read_csv(
        active_config.unlabelled_catalog_loc, 
        dtype={'id_str': str, 'label': int}
    )
    active_learning.run(
        unlabelled_catalog, 
        active_config.db_loc, 
        active_config.initial_size, 
        3,  # TODO channels not really generalized
        active_config.predictor_dir, 
        active_config.train_tfrecord_loc, 
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
