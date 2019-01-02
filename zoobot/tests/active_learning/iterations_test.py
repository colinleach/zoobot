import pytest

import os
import shutil
import time
import json

import numpy as np

from zoobot.active_learning import iterations
from zoobot.tests.active_learning import conftest


@pytest.fixture()
def initial_estimator_ckpt(tmpdir):
    return tmpdir.mkdir('some_datetime_ckpt').strpath


@pytest.fixture()
def new_iteration(tmpdir, initial_estimator_ckpt, active_config_ready):
        run_dir = active_config_ready.run_dir
        iteration_n = 0
        prediction_shards = ['first_shard.tfrecord', 'second_shard.tfrecord']

        iteration = iterations.Iteration(
            run_dir,
            iteration_n,
            prediction_shards,
            initial_db_loc=active_config_ready.db_loc,
            initial_train_tfrecords=[active_config_ready.shards.train_tfrecord_loc],
            train_callable=conftest.mock_train_callable,
            acquisition_func=conftest.mock_acquisition_func,
            n_samples=10,  # may need more samples?
            n_subjects_to_acquire=50,
            initial_size=64,
            initial_estimator_ckpt=None
        )

        return iteration


def test_init(tmpdir, initial_estimator_ckpt, active_config_ready):
        run_dir = active_config_ready.run_dir
        iteration_n = 0
        prediction_shards = ['some', 'shards']

        iteration = iterations.Iteration(
            run_dir,
            iteration_n,
            prediction_shards,
            initial_db_loc=active_config_ready.db_loc,
            initial_train_tfrecords=[active_config_ready.shards.train_tfrecord_loc],
            train_callable=np.random.rand,
            acquisition_func=np.random.rand,
            n_samples=10,  # may need more samples?
            n_subjects_to_acquire=50,
            initial_size=64,
            initial_estimator_ckpt=initial_estimator_ckpt
        )

        assert iteration.acquired_tfrecord is None
 
        expected_iteration_dir = os.path.join(run_dir, 'iteration_{}'.format(iteration_n))
        assert os.path.isdir(expected_iteration_dir)

        expected_estimators_dir = os.path.join(expected_iteration_dir, 'estimators')
        assert os.path.isdir(expected_estimators_dir)

        expected_metrics_dir = os.path.join(expected_iteration_dir, 'metrics')
        assert os.path.isdir(expected_metrics_dir)

        expected_db_loc = os.path.join(expected_iteration_dir, 'iteration.db')
        assert os.path.exists(expected_db_loc)

        # if initial estimator was provided, it should have been copied into the of 0th iteration subdir
        if initial_estimator_ckpt is not None:
            expected_initial_estimator_copy = os.path.join(expected_estimators_dir, 'some_datetime_ckpt')
            assert os.path.isdir(expected_initial_estimator_copy)


def test_get_latest_model(monkeypatch, new_iteration):
    def mock_get_latest_checkpoint(base_dir):
        return 'latest_ckpt'
    monkeypatch.setattr(iterations.active_learning, 'get_latest_checkpoint_dir', mock_get_latest_checkpoint)

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

    def mock_make_predictions_on_tfrecord(shard_locs, predictor, initial_size, n_samples):
        assert isinstance(shard_locs, list)
        assert predictor == 'loaded latest model'
        assert isinstance(initial_size, int)
        assert isinstance(n_samples, int)
        n_subjects = 256 * len(shard_locs)
        subjects = [{'matrix': np.random.rand(initial_size, initial_size, 3), 'id_str': str(n)} 
        for n in range(n_subjects)]
        samples = np.random.rand(n_subjects, n_samples)
        return subjects, samples
    monkeypatch.setattr(iterations.active_learning, 'make_predictions_on_tfrecord', mock_make_predictions_on_tfrecord)

    subjects, samples = new_iteration.make_predictions(shard_locs, size)
    assert len(subjects) == 256 * len(shard_locs)
    assert samples.shape == (256 * len(shard_locs), new_iteration.n_samples)


def test_save_metrics():
    pass  # does nothing except call external individually-unit-tested functions


def test_get_train_records(new_iteration, active_config_ready):
    assert new_iteration.get_train_records() == new_iteration.initial_train_tfrecords
    new_iteration.acquired_tfrecord = 'acquired.tfrecord'
    assert new_iteration.get_train_records() == new_iteration.initial_train_tfrecords + ['acquired.tfrecord']


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


def test_run(monkeypatch, new_iteration, previously_requested_subjects):
    SUBJECTS = [
        {'matrix': np.random.rand(new_iteration.initial_size, new_iteration.initial_size, 3),
        'id_str': str(n)}
        for n in range(1024)]
    
    SUBJECTS_REQUESTED = previously_requested_subjects.copy()  # may recieve random new subjects
    def mock_get_labels():
        selected_ids = SUBJECTS_REQUESTED.copy()
        selected_labels = list(np.random.rand(len(selected_ids)))
        SUBJECTS_REQUESTED.clear()
        assert len(SUBJECTS_REQUESTED) == 0
        return selected_ids, selected_labels
    monkeypatch.setattr(iterations, 'get_labels', mock_get_labels)
    def mock_request_labels(subject_ids):
        SUBJECTS_REQUESTED.extend(subject_ids)
    monkeypatch.setattr(iterations, 'request_labels', mock_request_labels)

    def mock_add_labelled_subjects_to_tfrecord(db, subject_ids, tfrecord_loc, size):
        assert len(subject_ids) > 0
        assert isinstance(subject_ids, list)
        assert os.path.exists(os.path.dirname(tfrecord_loc))
        assert isinstance(size, int)
        # save the subject ids here, pretending to be a tfrecord of those subjects
        with open(tfrecord_loc, 'w') as f:
            json.dump(subject_ids, f)
    monkeypatch.setattr(iterations.active_learning, 'add_labelled_subjects_to_tfrecord', mock_add_labelled_subjects_to_tfrecord)


    def mock_add_labels_to_db(subject_ids, labels, db):
        assert isinstance(subject_ids, list)
        assert isinstance(subject_ids[0], str)
        assert isinstance(labels, list)
        assert isinstance(labels[0], float)
        pass  # don't actually bother adding the new labels to the db
    monkeypatch.setattr(iterations.active_learning, 'add_labels_to_db', mock_add_labels_to_db)

    def mock_make_predictions(self, prediction_shards, initial_size):
        subjects = SUBJECTS[:len(prediction_shards) * 256] # imagining there are 256 subjects per shard
        samples = conftest.mock_get_samples_of_subjects(None, subjects, n_samples=self.n_samples)
        return subjects, samples
    monkeypatch.setattr(iterations.Iteration, 'make_predictions', mock_make_predictions)

    def mock_save_metrics(self, subjects, samples):
        assert os.path.isdir(self.metrics_dir)
        with open(os.path.join(self.metrics_dir, 'some_metrics.txt'), 'w') as f:
            f.write('some metrics from iteration {}'.format(self.name))
    monkeypatch.setattr(iterations.Iteration, 'save_metrics', mock_save_metrics)

    ####
    new_iteration.run()
    ####

    # previous iteration may have asked for some subjects - check they were acquired and used
    if len(previously_requested_subjects) > 0:
        assert new_iteration.acquired_tfrecord == os.path.join(new_iteration.requested_tfrecords_dir, 'acquired_shard.tfrecord')
        subjects_saved_from_earlier_request = json.load(open(new_iteration.acquired_tfrecord))
        assert subjects_saved_from_earlier_request == previously_requested_subjects
        assert new_iteration.acquired_tfrecord in new_iteration.get_train_records()
    else:
        assert new_iteration.acquired_tfrecord not in new_iteration.get_train_records()

    assert os.path.exists(os.path.join(new_iteration.metrics_dir, 'some_metrics.txt'))

    assert SUBJECTS_REQUESTED != previously_requested_subjects
    assert len(SUBJECTS_REQUESTED) == new_iteration.n_subjects_to_acquire
    subjects_acquired = [SUBJECTS[int(id_str)] for id_str in SUBJECTS_REQUESTED]
    subjects_means = np.array([subject['matrix'].mean() for subject in subjects_acquired])
    assert all(subjects_means[1:] < subjects_means[:-1])
