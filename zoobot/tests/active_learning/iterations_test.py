import pytest

import os
import shutil
import time
import json
import random

import numpy as np
import pandas as pd

from zoobot.active_learning import iterations, oracles
from zoobot.tests.active_learning import conftest


@pytest.fixture(params=[True, False])
def initial_estimator_ckpt(tmpdir, request):
    if request.param:
        return tmpdir.mkdir('some_datetime_ckpt').strpath
    else:
        return None  # no initial ckpt


@pytest.fixture()  # is actually also a test - the Iteration is real!
def new_iteration(monkeypatch, tmpdir, initial_estimator_ckpt, existing_db_loc, nonexistent_dir):
        iteration_dir = nonexistent_dir
        prediction_shards = ['first_shard.tfrecord', 'second_shard.tfrecord']
        def mock_get_db(iteration_dir, initial_db_loc):
            return 'connection to db'
        monkeypatch.setattr(iterations, 'get_db', mock_get_db)

        iteration = iterations.Iteration(
            iteration_dir,
            prediction_shards,
            initial_db_loc=existing_db_loc,
            initial_train_tfrecords=['train_a.tfrecord', 'train_b.tfrecord'],
            eval_tfrecords='shards.eval_tfrecord_locs()',
            train_callable=conftest.mock_train_callable,
            acquisition_func=conftest.mock_acquisition_func,
            n_samples=10,
            n_subjects_to_acquire=50,
            initial_size=64,
            initial_estimator_ckpt=initial_estimator_ckpt,
            learning_rate=0.001,
            epochs=2,
            oracle=oracles.Oracle()  # mocked above
        )

        assert not iteration.get_acquired_tfrecords()
 
        expected_iteration_dir = iteration_dir
        assert os.path.isdir(expected_iteration_dir)

        expected_estimators_dir = os.path.join(expected_iteration_dir, 'estimators')
        assert os.path.isdir(expected_estimators_dir)

        expected_metrics_dir = os.path.join(expected_iteration_dir, 'metrics')
        assert os.path.isdir(expected_metrics_dir)

        # removed because I mocked this    
        # expected_db_loc = os.path.join(expected_iteration_dir, 'iteration.db')
        # assert os.path.exists(expected_db_loc)

        # if initial estimator was provided, it should have been copied into the of 0th iteration subdir
        # TODO
        # if initial_estimator_ckpt is not None:
        #     expected_initial_estimator_copy = os.path.join(expected_estimators_dir, 'some_datetime_ckpt')
        #     assert os.path.isdir(expected_initial_estimator_copy)

        return iteration


def test_get_latest_model(monkeypatch, new_iteration):
    def mock_get_latest_checkpoint(base_dir):
        return 'latest_ckpt'
    monkeypatch.setattr(iterations.misc, 'get_latest_checkpoint_dir', mock_get_latest_checkpoint)

    def mock_load_predictor(predictor_loc):
        if predictor_loc == 'latest_ckpt':
            return 'loaded latest model'
        else:
            return 'loaded another model'
    monkeypatch.setattr(iterations.make_predictions, 'load_predictor', mock_load_predictor)

    assert new_iteration.get_latest_model() == 'loaded latest model'


def test_make_predictions(monkeypatch, shard_locs, size, new_iteration):
    def mock_get_latest_model(self):
        return 'loaded latest model'
    monkeypatch.setattr(iterations.Iteration, 'get_latest_model', mock_get_latest_model)

    def mock_make_predictions_on_tfrecord(shard_locs, predictor, db, n_samples, size):
        assert isinstance(shard_locs, list)
        assert predictor == 'loaded latest model'
        assert isinstance(size, int)
        assert isinstance(n_samples, int)
        n_subjects = 112 * len(shard_locs)  # 112 unlabelled subjects per shard
        subjects = [{'matrix': np.random.rand(size, size, 3), 'id_str': str(n)} 
        for n in range(n_subjects)]
        samples = np.random.rand(n_subjects, n_samples)
        return subjects, samples
    monkeypatch.setattr(iterations.database, 'make_predictions_on_tfrecord', mock_make_predictions_on_tfrecord)

    subjects, samples = new_iteration.make_predictions(shard_locs, size)
    assert len(subjects) == 112 * len(shard_locs)
    assert samples.shape == (112 * len(shard_locs), new_iteration.n_samples)


def test_get_train_records(new_iteration):
    assert new_iteration.get_train_records() == new_iteration.initial_train_tfrecords
    tfrecord_loc = os.path.join(new_iteration.acquired_tfrecords_dir, 'something.tfrecord')
    with open(tfrecord_loc, 'w') as f:
        f.write('a mock tfrecord')
    assert new_iteration.get_train_records() == new_iteration.initial_train_tfrecords + [tfrecord_loc]


def test_record_train_records(new_iteration):
    new_iteration.record_train_records()
    with open(os.path.join(new_iteration.iteration_dir, 'train_records_index.json'), 'r') as f:
        train_records = json.load(f)
    assert train_records == new_iteration.get_train_records()


@pytest.fixture(params=[False, True])
def previously_requested_subjects(request, new_iteration):
    if request.param:  # previous iteration has picked random subjects to be acquired
        return [str(n) for n in range(new_iteration.n_subjects_to_acquire)]
    else:
        return []


def test_run(mocker, monkeypatch, new_iteration, previously_requested_subjects):
    SUBJECTS = [
        {'matrix': np.random.rand(new_iteration.initial_size, new_iteration.initial_size, 3),
        'id_str': str(n)}
        for n in range(1024)]
    
    # TODO move this to oracle tests
    class MockOracle():
        def __init__(self):
            self.subjects_requested = previously_requested_subjects.copy()  # may recieve random new subjects
        def get_labels(self, labels_dir): # should really be all labels and then filter. Instead, filter here and skip filter later.
            selected_ids = self.subjects_requested.copy()
            label_col_a = list(np.random.rand(len(selected_ids)))
            label_col_b = [40 for n in range(len(label_col_a))]
            labels = [{'label_a': a, 'label_b': b} for (a, b) in zip(label_col_a, label_col_b)]
            selected_labels = labels.copy()  # see above, messy all/filtering
            self.subjects_requested.clear()
            assert len(self.subjects_requested) == 0
            return selected_ids, selected_labels
        def request_labels(self, subject_ids, name, retirement):
            self.subjects_requested.extend(subject_ids)
    new_iteration.oracle = MockOracle()

    def mock_filter_for_new_only(db, subjects, labels): # skip filter later
        return subjects, labels
    monkeypatch.setattr(iterations.database, 'filter_for_new_only', mock_filter_for_new_only)

    # still needed?
    # def mock_get_file_loc_df_from_db(db, subject_ids):
    #     data = {
    #         'id_str': subject_ids,
    #         'file_loc': 'somewhere',
    #         'label': np.random.randint(low=0, high=40, size=len(subject_ids))
    #     }
    #     return pd.DataFrame(data=data)
    # monkeypatch.setattr(iterations.active_learning, 'get_file_loc_df_from_db', mock_get_file_loc_df_from_db)
    
    def mock_get_specific_subjects(db, subject_ids):
        data = [{'id_str': id_str, 'file_loc': 'somewhere', 'labels': {'label_a': 'a', 'label_b': 'b'}} for id_str in subject_ids]
        return pd.DataFrame(data=data)
    monkeypatch.setattr(iterations.database, 'get_specific_subjects', mock_get_specific_subjects)

    def mock_write_catalog_to_tfrecord_shards(df, db, img_size, columns_to_save, save_dir, shard_size):
        # save the subject ids here, pretending to be a tfrecord of those subjects
        assert set(columns_to_save) == {'id_str', 'labels'}
        tfrecord_loc = os.path.join(save_dir, 'shard_0.tfrecord')
        with open(tfrecord_loc, 'w') as f:
            json.dump([str(x) for x in df['id_str']], f)
    monkeypatch.setattr(iterations.database, 'write_catalog_to_tfrecord_shards', mock_write_catalog_to_tfrecord_shards)


    def mock_add_labelled_subjects_to_tfrecord(db, subject_ids, tfrecord_loc, size):
        assert len(subject_ids) > 0
        assert isinstance(subject_ids, list)
        assert os.path.exists(os.path.dirname(tfrecord_loc))
        assert isinstance(size, int)
        # save the subject ids here, pretending to be a tfrecord of those subjects
        with open(tfrecord_loc, 'w') as f:
            json.dump(subject_ids, f)
    monkeypatch.setattr(iterations.database, 'add_labelled_subjects_to_tfrecord', mock_add_labelled_subjects_to_tfrecord)


    def mock_add_labels_to_db(subject_ids, labels, db):
        assert isinstance(subject_ids, list)
        assert isinstance(subject_ids[0], str)
        assert isinstance(labels, list)
        assert isinstance(labels[0], dict)
        pass  # don't actually bother adding the new labels to the db
        # TODO use a mock and check the call for ids and labels
    monkeypatch.setattr(iterations.db_access, 'add_labels_to_db', mock_add_labels_to_db)

    def mock_make_predictions(self, prediction_shards, initial_size):
        subjects = SUBJECTS[:len(prediction_shards) * 256]  # imagining there are 256 subjects per shard
        unlabelled_subjects = random.sample(subjects, 212)  # some of which are labelled
        images = np.array([subject['matrix'] for subject in unlabelled_subjects])
        samples = conftest.mock_get_samples_of_images(None, images, n_samples=self.n_samples)
        return unlabelled_subjects, samples
    monkeypatch.setattr(iterations.Iteration, 'make_predictions', mock_make_predictions)

    def mock_db_fully_labelled(db):
        return False
    monkeypatch.setattr(iterations.database, 'db_fully_labelled', mock_db_fully_labelled)

    # TODO check for single call with correct attrs?
    # mocker.Mock('zoobot.active_learning.iterations.Iteration.record_state')

    ####
    new_iteration.run()
    ####

    # TODO review
    # check that the initial ckpt was copied successfully, if one was given
    if new_iteration.initial_estimator_ckpt is not None:
        expected_ckpt_copy = os.path.join(new_iteration.estimators_dir, new_iteration.initial_estimator_ckpt)
        assert os.path.isdir(expected_ckpt_copy)

    # previous iteration may have asked for some subjects - check they were acquired and used
    expected_tfrecords = [
        os.path.join(new_iteration.acquired_tfrecords_dir, 'shard_0.tfrecord')
    ]
    if previously_requested_subjects:
        assert new_iteration.get_acquired_tfrecords() == expected_tfrecords
        # mock_write_catalog_to_tfrecord_shards saved json of acquired id_strs to each tfrecord
        subjects_saved_from_earlier_request = json.load(open(expected_tfrecords[0]))
        assert subjects_saved_from_earlier_request == previously_requested_subjects
        assert set(expected_tfrecords).issubset(set(new_iteration.get_train_records()))
    else:
        assert set(expected_tfrecords).intersection(set(new_iteration.get_train_records())) == set()

    # check records were saved TODO
    # assert os.path.exists(os.path.join(new_iteration.metrics_dir, 'some_metrics.txt'))

    # check the correct subjects were requested
    assert new_iteration.oracle.subjects_requested != previously_requested_subjects
    assert len(new_iteration.oracle.subjects_requested) == new_iteration.n_subjects_to_acquire
    subjects_acquired = [SUBJECTS[int(id_str)] for id_str in new_iteration.oracle.subjects_requested]
    subjects_means = np.array([subject['matrix'].mean() for subject in subjects_acquired])
    assert all(subjects_means[1:] < subjects_means[:-1])
