import os
import shutil
import logging
import json
import sqlite3
from typing import List

import numpy as np

from zoobot.estimators import make_predictions, losses
from zoobot.active_learning import database, db_access, metrics, misc, run_estimator_config


class Iteration():

    def __init__(
        self,
        iteration_dir: str,
        prediction_shards: List,
        initial_db_loc: str,
        initial_train_tfrecords: List,
        eval_tfrecords: List,
        fixed_estimator_params,
        acquisition_func,
        n_samples: int,  # may need more samples?
        n_subjects_to_acquire: int,
        initial_size: int,
        learning_rate: float,
        epochs: int,
        oracle,
        questions,
        label_cols,
        initial_estimator_ckpt=None
    ):
        """
        Do some sanity checks in the args, then save them as properties.

        Using iteration_dir, create the directory tree needed:
        - {iteration_dir}
            - acquired_tfrecords
            - metrics
            - estimators
        Copy the estimator from initial_estimator_ckpt, if provided.

        Copy the database from initial_db_loc to {iteration_dir}/iteration.db, and connect.

        Prepare to record the tfrecords used under {iteration_dir}/train_tfrecords_index.json
        """

        self.iteration_dir = iteration_dir

        # shards should be unique, or everything falls apart.
        if not len(prediction_shards) == len(set(prediction_shards)):
            raise ValueError(
                'Warning: duplicate prediction shards! {}'.format(prediction_shards))
        self.prediction_shards = prediction_shards

        for (tfrecords, attr) in [
            # acquired up to start of iteration
            (initial_train_tfrecords, 'initial_train_tfrecords'),
                (eval_tfrecords, 'eval_tfrecords')]:
            assert isinstance(initial_train_tfrecords, list)
            try:
                assert all([os.path.isfile(loc)
                            for loc in initial_train_tfrecords])
            except AssertionError:
                logging.critical('Fatal error: missing {}!'.format(attr))
                logging.critical(tfrecords)
            setattr(self, attr, tfrecords)

        # assert callable(fixed_estimator_params)
        self.fixed_estimator_params = fixed_estimator_params
        assert callable(acquisition_func)
        self.acquisition_func = acquisition_func
        self.n_samples = n_samples
        self.n_subjects_to_acquire = n_subjects_to_acquire
        # need to know what size to write new images to shards
        self.initial_size = initial_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.oracle = oracle
        
        self.estimators_dir = os.path.join(self.iteration_dir, 'estimators')
        self.acquired_tfrecords_dir = os.path.join(
            self.iteration_dir, 'acquired_tfrecords')
        self.labels_dir = os.path.join(self.iteration_dir, 'acquired_labels')
        self.metrics_dir = os.path.join(self.iteration_dir, 'metrics')

        if os.path.isdir(self.iteration_dir):
            logging.warning(f'{self.iteration_dir} already exists - deleting!')
            shutil.rmtree(self.iteration_dir)
        os.mkdir(self.iteration_dir)
        os.mkdir(self.acquired_tfrecords_dir)
        os.mkdir(self.metrics_dir)
        # TODO have a test that verifies new folder structure?
        
        # decals schema for now, but will likely switch to GZ2
        self.schema = losses.Schema(self.fixed_estimator_params.label_cols, self.fixed_estimator_params.questions, version='gz2')

        self.run_config = run_estimator_config.get_run_config(
            initial_size=self.fixed_estimator_params.initial_size, 
            final_size=self.fixed_estimator_params.final_size,
            schema=self.schema,
            batch_size=self.fixed_estimator_params.batch_size,
            warm_start=False,  # for now 
            log_dir=self.estimators_dir,
            train_records=self.get_train_records(),
            eval_records=self.eval_tfrecords,  # linting error due to __init__, self.eval_tfrecords exists
            learning_rate=self.learning_rate,
            epochs=self.epochs,

        )  # could be docker container to run, save model

        self.db, self.db_loc = get_db(self.iteration_dir, initial_db_loc)

        self.initial_estimator_ckpt = initial_estimator_ckpt

        src = self.initial_estimator_ckpt
        dest = self.estimators_dir
        if initial_estimator_ckpt is not None:
            logging.info('Copying {} initial estimator ckpt'.format(
                initial_estimator_ckpt))

            # copy the initial estimator folder inside estimators_dir, keeping the same name
            shutil.copytree(
                src=src,
                dst=dest
            )

            # copy the files only, subdirs are saved models
            # [shutil.copy(f, dest) for f in os.listdir(src) if os.path.isfile(f)]

            # remove this log from the copy, to save space
            # may have changed in TF2?
            [os.remove(os.path.join(dest, f)) for f in os.listdir(
                dest) if f.startswith('events.out.tfevents')]
        else:
            os.mkdir(self.estimators_dir)

        # record which tfrecords were used, for later analysis
        self.tfrecords_record = os.path.join(
            self.iteration_dir, 'train_records_index.json')

    def get_acquired_tfrecords(self):
        return [os.path.join(self.acquired_tfrecords_dir, loc) for loc in os.listdir(self.acquired_tfrecords_dir)]

    def get_train_records(self):
        return self.initial_train_tfrecords + self.get_acquired_tfrecords()  # linting error

    def make_predictions(self, model):
        # logging.debug('Loaded predictor {}'.format(predictor))
        logging.info('Making and recording predictions')
        logging.info('Using shard_locs {}'.format(self.prediction_shards))
        unlabelled_subjects, samples = database.make_predictions_on_tfrecord(
            self.prediction_shards,
            model,
            self.run_config,
            self.db,
            n_samples=self.n_samples,
            size=self.initial_size
        )
        # subjects should all be unique, otherwise there's a bug
        id_strs = [subject['id_str'] for subject in unlabelled_subjects]
        assert len(id_strs) == len(set(id_strs))
        assert isinstance(unlabelled_subjects, list)
        return unlabelled_subjects, samples

    def record_state(self, subjects, samples, acquisitions):
        metrics.save_iteration_state(
            self.iteration_dir, subjects, samples, acquisitions)

    def run(self):
        """
        Actually run an active learning step!

        Get the latest labels from the oracle
        
        Record the train tfrecords used
        Execute train_callable (i.e. train the model)

        Make new predictions on the unlabelled shards
        From the predictions, calculate acquisition function values
        Record both the predictions and acquisition values to the database
        Get the id's of the unlabelled subjects with the highest acquisition values,
        and request labels for them.
        """
        all_subject_ids, all_labels = self.oracle.get_labels(
            self.labels_dir)
        # can't allow overwriting of previous labels, as may have been written to tfrecord
        if len(all_subject_ids) > 0:
            subject_ids, labels = database.filter_for_new_only(
                self.db,
                all_subject_ids,
                all_labels
            )
            if len(subject_ids) > 0:
                # record in db
                # TODO should be via database.py?
                db_access.add_labels_to_db(subject_ids, labels, self.db)
                # write to disk
                top_subject_df = database.get_specific_subjects_df(
                    self.db, subject_ids)
                database.write_catalog_to_tfrecord_shards(
                    top_subject_df,
                    db=None,  # don't record again in db as simply a fixed train record
                    img_size=self.initial_size,
                    columns_to_save=['id_str'] + self.fixed_estimator_params.label_cols,
                    save_dir=self.acquired_tfrecords_dir,
                    shard_size=4096  # hardcoded, awkward TODO
                )
            else:
                logging.warning('All acquired subjects are already labelled - does this make sense?')
            assert not database.db_fully_labelled(self.db)
        else:
            logging.warning('No subjects have been returned from the oracle - does this make sense?')

        """Callable should expect 
        - log dir to train models in
        - list of tfrecord files to train on
        - list of tfrecord files to eval on
        - learning rate to use 
        - epochs to train for"""
        self.record_train_records()
        logging.info('Saving to {}'.format(self.estimators_dir))

        best_model = self.run_config.run_estimator()  # saves weights to {estimator_dir i.e. log_dir}/models/final

        # TODO getting quite messy throughout with lists vs np.ndarray - need to clean up
        # make predictions and save to db, could be docker container
        subjects, predictions = self.make_predictions(best_model)

        # returns list of acquisition values
        # warning, duplication
        
        acquisitions = self.acquisition_func(predictions, self.schema, retirement=40)
        self.record_state(subjects, predictions, acquisitions)
        logging.debug('{} {} {}'.format(
            len(acquisitions), len(subjects), len(predictions)))

        _, top_acquisition_ids = pick_top_subjects(
            subjects, acquisitions, self.n_subjects_to_acquire)
        self.oracle.request_labels(
            top_acquisition_ids, name='priority', retirement=40)

    def record_train_records(self):
        with open(os.path.join(self.tfrecords_record), 'w') as f:
            json.dump(self.get_train_records(), f)


# to be shared for consistency
def pick_top_subjects(subjects, acquisitions, n_subjects_to_acquire):
    # reverse order, highest to lowest
    args_to_sort = np.argsort(acquisitions)[::-1]
    top_acquisition_subjects = [subjects[i]
                                for i in args_to_sort][:n_subjects_to_acquire]
    top_acquisition_ids = [subject['id_str']
                           for subject in top_acquisition_subjects]
    assert len(top_acquisition_ids) == len(
        set(top_acquisition_ids))  # no duplicates allowed
    return top_acquisition_subjects, top_acquisition_ids


def get_db(iteration_dir, initial_db_loc):
    new_db_loc = os.path.join(iteration_dir, 'iteration.db')
    assert os.path.isfile(initial_db_loc)
    shutil.copy(initial_db_loc, new_db_loc)
    db = sqlite3.connect(new_db_loc)
    assert not database.db_fully_labelled(db)
    return db, new_db_loc
