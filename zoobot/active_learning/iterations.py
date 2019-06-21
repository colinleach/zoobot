import os
import shutil
import logging
import json
import sqlite3

import numpy as np

from zoobot.estimators import make_predictions
from zoobot.active_learning import mock_panoptes
from zoobot.active_learning import active_learning, metrics, acquisition_utils


class Iteration():

    def __init__(
        self, 
        run_dir,
        iteration_n,
        prediction_shards,
        initial_db_loc,
        initial_train_tfrecords,
        train_callable,
        acquisition_func,
        n_samples,  # may need more samples?
        n_subjects_to_acquire,
        initial_size,
        initial_estimator_ckpt=None
        ):

        self.name = 'iteration_{}'.format(iteration_n)
        # shards should be unique, or everything falls apart.
        assert len(prediction_shards) == len(set(prediction_shards))
        self.prediction_shards = prediction_shards
        assert isinstance(initial_train_tfrecords, list)
        self.initial_train_tfrecords = initial_train_tfrecords  # acquired up to start of iteration
        self.acquired_tfrecord = None
        assert callable(train_callable)
        self.train_callable = train_callable
        assert callable(acquisition_func)
        self.acquisition_func = acquisition_func
        self.n_samples = n_samples
        self.n_subjects_to_acquire = n_subjects_to_acquire
        self.initial_size = initial_size  # need to know what size to write new images to shards

        self.iteration_dir = os.path.join(run_dir, self.name)
        self.estimators_dir = os.path.join(self.iteration_dir, 'estimators')
        self.requested_tfrecords_dir = os.path.join(self.iteration_dir, 'requested_tfrecords')
        self.metrics_dir = os.path.join(self.iteration_dir, 'metrics')

        os.mkdir(self.iteration_dir)
        os.mkdir(self.estimators_dir)
        os.mkdir(self.requested_tfrecords_dir)
        os.mkdir(self.metrics_dir)

        self.db_loc = os.path.join(self.iteration_dir, 'iteration.db')
        shutil.copy(initial_db_loc, self.db_loc)
        self.db = sqlite3.connect(self.db_loc)

        if initial_estimator_ckpt is not None:
            # copy the initial estimator folder inside estimators_dir, keeping the same name
            shutil.copytree(
                src=initial_estimator_ckpt, 
                dst=os.path.join(self.estimators_dir, os.path.split(initial_estimator_ckpt)[-1])
            )

        # record which tfrecords were used, for later analysis
        self.tfrecords_record = os.path.join(self.iteration_dir, 'train_records_index.json')


    def get_train_records(self):
        if self.acquired_tfrecord is None:
            return self.initial_train_tfrecords
        else:
            return self.initial_train_tfrecords + [self.acquired_tfrecord]


    def make_predictions(self, shard_locs, initial_size):
        predictor = self.get_latest_model()
        logging.info('Making and recording predictions')
        logging.info('Using shard_locs {}'.format(shard_locs))
        subjects, samples = active_learning.make_predictions_on_tfrecord(shard_locs, predictor, initial_size=initial_size, n_samples=self.n_samples)
        # subjects should all be unique, otherwise there's a bug
        id_strs = [subject['id_str'] for subject in subjects]
        assert len(id_strs) == len(set(id_strs)) 
        return subjects, samples


    def get_latest_model(self):
        predictor_loc = active_learning.get_latest_checkpoint_dir(self.estimators_dir)
        logging.info('Loading model from {}'.format(predictor_loc))
        return make_predictions.load_predictor(predictor_loc)


    def save_metrics(self, subject_data, samples):
        """[summary]
        
        Args:
            subject_data (np.array): of form [n_subjects, height, width, channels]. NOT a list.
            samples (np.array): model predictions for rho, of form [n_subjects, n_samples].
        """

        
        # TODO allow for direct acquisitions passing, for speed?
        # TODO check entropies against radial extent of galaxy
        # TODO add metrics for each active learning run, cross-matching to catalog for NSA params via id
        model = metrics.Model(samples, labels=None, name=self.name)
        model.show_acquisitions_vs_predictions(save_dir=self.metrics_dir)
        acquisition_utils.save_acquisition_examples(
            subject_data, 
            model.mutual_info, 
            acq_string='mutual_info', 
            save_dir=self.metrics_dir
        )
        # TODO still need to verify that acq function values match up, or pass them directly


    def run(self):
        subject_ids, labels = get_labels()
        if len(subject_ids) > 0:
            active_learning.add_labels_to_db(subject_ids, labels, self.db)
            self.acquired_tfrecord = os.path.join(self.requested_tfrecords_dir, 'acquired_shard.tfrecord')
            active_learning.add_labelled_subjects_to_tfrecord(self.db, subject_ids, self.acquired_tfrecord, self.initial_size)

        # callable should expect 
        # - log dir to train models in
        # - list of tfrecord files to train on
        self.record_train_records()
        self.train_callable(self.estimators_dir, self.get_train_records())  # could be docker container to run, save model

        # make predictions and save to db, could be docker container
        subjects, samples = self.make_predictions(self.prediction_shards, self.initial_size)

        acquisitions = self.acquisition_func(samples)  # returns list of acquisition values
        self.save_metrics(subjects, samples)

        args_to_sort = np.argsort(acquisitions)[::-1]  # reverse order, highest to lowest
        top_acquisition_subjects = [subjects[i] for i in args_to_sort][:self.n_subjects_to_acquire]
        top_acquisition_ids = [subject['id_str'] for subject in top_acquisition_subjects]
        assert len(top_acquisition_ids) == len(set(top_acquisition_ids))  # no duplicates allowed
        # TODO save acquisition ids for posterity?

        request_labels(top_acquisition_ids)


    def record_train_records(self):
        with open(os.path.join(self.tfrecords_record), 'w') as f:
            json.dump(self.get_train_records(), f)


def request_labels(top_acquisition_ids):
    mock_panoptes.request_labels(top_acquisition_ids)


def get_labels():
    return mock_panoptes.get_labels()



    # def get_train_records(self):
    #     logging.info('Attempting to load {}'.format(self.train_records_index_loc))
    #     with open(self.train_records_index_loc, 'r') as f:  # must exist, see __init__
    #         train_records = json.load(f)  # restore from disk all previous train records
    #     logging.info('Loaded train records: {}'.format(train_records))
    #     assert isinstance(train_records, list)
    #     return train_records


    # def add_train_record(self, new_record_loc):
    #     # must always be kept in sync
    #     current_records = self.get_train_records()
    #     current_records.append(new_record_loc)
    #     self.write_train_records_index(current_records)

