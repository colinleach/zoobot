import os
import shutil
import logging
import json
import sqlite3
from typing import List
import glob

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
        n_samples: int,
        n_subjects_to_acquire: int,
        initial_size: int,
        learning_rate: float,
        epochs: int,
        oracle,
        questions: List,
        label_cols: List,
        checkpoints=[]
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

        if not all([os.path.isfile(loc) for loc in prediction_shards]):
            raise ValueError('Missing some prediction shards, of {}'.format(prediction_shards))
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
        # print(self.n_subjects_to_acquire)
        # exit()
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

        self.db, self.db_loc = get_db(self.iteration_dir, initial_db_loc)
        
        os.mkdir(self.estimators_dir)

        assert len(checkpoints) > 0  # should not be empty list
        self.checkpoints = checkpoints # train models loaded from these checkpoints. Skip training if 'pretrained' in name.
        self.prediction_checkpoints = []  # make predictions using models loaded from these checkpoints. 


        # record which tfrecords were used, for later analysis
        self.tfrecords_record = os.path.join(
            self.iteration_dir, 'train_records_index.json')

    def run_config(self, weights_loc, log_dir):
        return run_estimator_config.get_run_config(
            initial_size=self.fixed_estimator_params.initial_size, 
            final_size=self.fixed_estimator_params.final_size,
            crop_size=self.fixed_estimator_params.crop_size,
            schema=self.schema,
            batch_size=self.fixed_estimator_params.batch_size,
            weights_loc=weights_loc,  # for now 
            log_dir=log_dir,
            train_records=self.get_train_records(),  # will always be up-todate
            eval_records=self.eval_tfrecords,  # linting error due to __init__, self.eval_tfrecords exists
            learning_rate=self.learning_rate,
            epochs=self.epochs,
        )  # could be docker container to run, save model

    def get_acquired_tfrecords(self):
        return [os.path.join(self.acquired_tfrecords_dir, loc) for loc in os.listdir(self.acquired_tfrecords_dir)]

    def get_train_records(self):
        return self.initial_train_tfrecords + self.get_acquired_tfrecords()  # linting error

    # supports using only n trained models *from this iteration*, but probably don't need this now
    def get_prediction_models(self, max_models=10):
        if self.prediction_checkpoints == []:
            logging.critical('No previous prediction checkpoints found - multimodel acquisition will not work!')
        if max_models < len(self.prediction_checkpoints):
            logging.info('Loading only {} of {} predictors'.format(max_models, len(self.prediction_checkpoints)))
            checkpoints = self.prediction_checkpoints[-max_models::]
        else:
            checkpoints = self.prediction_checkpoints
        logging.info(f'Loading predictors: {checkpoints}')
        models = []
        for loc in checkpoints:
            models.append(run_estimator_config.get_model(
                self.schema,
                self.fixed_estimator_params.initial_size,
                self.fixed_estimator_params.crop_size,
                self.fixed_estimator_params.final_size,
                weights_loc=loc)
            )
        return models

    def make_predictions(self, model):
        logging.info('Making and recording predictions')
        logging.info('Using shard_locs {}'.format(self.prediction_shards))
        unlabelled_subjects, samples = database.make_predictions_on_tfrecord(
            self.prediction_shards,
            model,
            self.fixed_estimator_params,
            self.db,
            n_samples=self.n_samples,
            size=self.initial_size
        )
        # subjects should all be unique, otherwise there's a bug
        id_strs = [subject['id_str'] for subject in unlabelled_subjects]
        assert len(id_strs) == len(set(id_strs))
        assert isinstance(unlabelled_subjects, list)
        return unlabelled_subjects, samples

    def record_state(self, subjects, samples: List, acquisitions):
        metrics.save_iteration_state(
            self.iteration_dir, subjects, samples, acquisitions)

    def run(self):
        """
        Actually run an active learning step!

        Get the latest labels from the oracle
        
        Record the train tfrecords used
        Train the model

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

        # this may have added new train records - make sure run_config is using them. 
        # self.run_config will always use the latest as it is recreated each time via @property

        """Callable should expect 
        - log dir to train models in
        - list of tfrecord files to train on
        - list of tfrecord files to eval on
        - learning rate to use 
        - epochs to train for"""
        self.record_train_records()
        logging.info('Saving to {}'.format(self.estimators_dir))

        if self.prediction_checkpoints:
            logging.critical('Expected no prediction checkpoints yet, got {}'.format(self.prediction_checkpoints))
        for checkpoint_n, checkpoint_loc in enumerate(self.checkpoints): # may be [None, None]

            # maybe we can skip training?
            skip = False
            if checkpoint_loc is None:
                weights_folder_name = 'model_{}'.format(checkpoint_n)
            else:
                weights_folder_name = os.path.basename(os.path.dirname(checkpoint_loc))
                if 'pretrained' in checkpoint_loc:
                    skip = True

            target_dir = os.path.join(self.estimators_dir, 'models', weights_folder_name)  # will have final.index or in_progress.index etc in this folder
            log_dir = os.path.join(target_dir, 'log')  # for tensorboard etc e.g. models/model_a/log/blah.tfevents.out
            checkpoint_name = 'final'
            if 'in_progress' in checkpoint_loc:  # checkpint to load is actually called in_progress
                checkpoint_name = 'in_progress'
            save_loc = os.path.join(target_dir, checkpoint_name)  # weights files e.g. models/model_a/final.index. Do not attempt to rename in case of copy (skip, below)
    
            if skip:
                logging.warning('Skipping training and loading cheat estimators from {}'.format(checkpoint_loc))
                # copy the existing weights, mimicking saving the model

                current_weights_folder = os.path.dirname(checkpoint_loc + '.index')
                shutil.copytree(current_weights_folder, target_dir)  # contents of folder a into new folder called folder b
                # logging.debug('Copied files: {}'.format(glob.glob(target_dir + '/*')))
                logging.info('Copied pretrained weights from {} to {} to mimick training'.format(current_weights_folder, target_dir))
            else:
                logging.info('No skip estimators found at {} - beginning training'.format(checkpoint_loc))
                trained_model = self.run_config(checkpoint_loc, log_dir=log_dir).run_estimator()
                logging.info('Saving trained model to {}'.format(save_loc))
                trained_model.save_weights(save_loc)  # may overwrite the best model (in-progress) saved by run_estimator()?

            logging.info('Expecting saved weights at {}.index'.format(save_loc))
            assert os.path.isfile(save_loc + '.index')  # either from training, or from copytree (note that the checkpoint name must remain constant)
            self.prediction_checkpoints.append(save_loc)
        logging.info('All training completed. Prediction checkpoints to use: {}'.format(self.prediction_checkpoints))

        # TODO getting quite messy throughout with lists vs np.ndarray - need to clean up
        # make predictions and save to db, could be docker container
        subject_ids = None
        all_predictions = []
        for model_n, model in enumerate(self.get_prediction_models()): # N most recent models
            logging.info(f'Predicting with model {model_n}')
            subjects, predictions = self.make_predictions(model)
            if subject_ids is not None:
                # double check that subjects remains consistent
                new_subject_ids = [s['id_str'] for s in subjects]
                assert all([s_new == s_old for (s_new, s_old) in zip(subject_ids, new_subject_ids)])
                subject_ids = new_subject_ids  # for next time
            all_predictions.append(predictions)
            # all_predictions.append(predictions[:100])  # TEMP to debug
        logging.info('All model predictions: {}'.format([p.shape for p in all_predictions]))

        # returns list of acquisition values
        # warning, duplication
        acquisitions = self.acquisition_func(all_predictions, self.schema)
        logging.info('Acquistions: ', acquisitions)
        logging.info(acquisitions.shape)
        if acquisitions.ndim > 1:
            logging.critical('Acquisitions ndim > 1: you probably forgot to take a mean per question/answer?')

        logging.info('{} {} {}'.format(
            len(acquisitions), len(subjects), len(predictions)))

        self.record_state(subjects, all_predictions, acquisitions)

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
    logging.info(n_subjects_to_acquire)
    logging.info(args_to_sort[:10])
    top_acquisition_subjects = [subjects[i]
                                for i in args_to_sort][:n_subjects_to_acquire]
    top_acquisition_ids = [subject['id_str']
                           for subject in top_acquisition_subjects]
    logging.info(top_acquisition_ids[:10])
    assert len(top_acquisition_ids) == len(
        set(top_acquisition_ids))  # no duplicates allowed
    return top_acquisition_subjects, top_acquisition_ids


def get_db(iteration_dir, initial_db_loc, timeout=15.0):
    new_db_loc = os.path.join(iteration_dir, 'iteration.db')
    assert os.path.isfile(initial_db_loc)
    shutil.copy(initial_db_loc, new_db_loc)
    db = sqlite3.connect(new_db_loc, timeout=timeout, isolation_level='IMMEDIATE')
    assert not database.db_fully_labelled(db)
    return db, new_db_loc
