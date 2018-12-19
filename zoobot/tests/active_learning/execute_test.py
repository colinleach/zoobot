import os
import json
import time
import copy

import pytest
import numpy as np

from zoobot.tfrecord import read_tfrecord
from zoobot.active_learning import active_learning, make_shards, execute



def test_prepare_run_folders():
    # TODO
    pass



def test_run(active_config_ready, tmpdir, monkeypatch, catalog_random_images, tfrecord_dir, acquisition_func):
    assert active_config_ready.ready()
    catalog = catalog_random_images  # fits files must really exist

    # depends on make_database_and_shards working okay
    make_shards.make_database_and_shards(catalog, active_config_ready.db_loc, active_config_ready.shards.initial_size, tfrecord_dir, shard_size=25)

    def train_callable(train_tfrecord_locs):
        # pretend to save a model in subdirectory of estimator_dir
        subdir_loc = os.path.join(active_config_ready.estimator_dir, str(time.time()))
        os.mkdir(subdir_loc)

    def mock_load_predictor(loc):
        return None
    monkeypatch.setattr(active_learning.make_predictions, 'load_predictor', mock_load_predictor)

    def mock_get_labels(subject_ids):  # don't actually read from saved catalog, just make up
        return [np.random.randint(2) for n in range(len(subject_ids))]
    monkeypatch.setattr(active_learning.mock_panoptes, 'get_labels', mock_get_labels)

    def get_acquistion_func(predictor):
        return acquisition_func

    # TODO add something else (time, string) in predictor dir and make sure the latest timestamp is loaded
    active_config_ready.run(train_callable, get_acquistion_func)
    # TODO instead of blindly cycling through shards, record where the shards are and latest update

    # read back the training tfrecords and verify they are sorted by order of mean
    with open(active_config_ready.train_records_index_loc, 'r') as f:
        training_shards = json.load(f)[1:]  # includes the initial shard, which is unsorted
    
    for shard in training_shards:
        subjects = read_tfrecord.load_examples_from_tfrecord(
            [shard], 
            read_tfrecord.matrix_label_id_feature_spec(active_config_ready.shards.initial_size, active_config_ready.shards.channels)
        )
        matrix_means = np.array([x['matrix'].mean() for x in subjects])
        assert np.all(matrix_means[1:] < matrix_means[:-1])  # monotonically decreasing (highest written first)
