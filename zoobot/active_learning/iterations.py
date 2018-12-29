import os
import shutil
import logging

import numpy as np

from zoobot.estimators import make_predictions
from zoobot.active_learning import active_learning
from zoobot.uncertainty import check_uncertainty


class Iteration():

    def __init__(
        self, 
        run_dir,
        iteration_n,
        initial_estimator_ckpt=None
        ):

        self.iteration_dir = os.path.join(run_dir, 'iteration_{}'.format(iteration_n))
        self.estimators_dir = os.path.join(self.iteration_dir, 'estimators')
        self.metrics_dir = os.path.join(self.iteration_dir, 'metrics')

        os.mkdir(self.iteration_dir)
        os.mkdir(self.estimators_dir)

        if initial_estimator_ckpt is not None:
            # copy the initial estimator folder inside estimators_dir, keeping the same name
            shutil.copytree(
                src=initial_estimator_ckpt, 
                dst=os.path.join(self.estimators_dir, os.path.split(initial_estimator_ckpt)[-1])
            )


    def make_predictions(self, shard_locs, initial_size):
        predictor = self.get_latest_model()
        logging.info('Making and recording predictions')
        logging.info('Using shard_locs {}'.format(shard_locs))
        subjects, samples = active_learning.make_predictions_on_tfrecord(shard_locs, predictor, initial_size=initial_size, n_samples=20) # may need more samples?
        return subjects, samples


    def get_latest_model(self):
            predictor_loc = active_learning.get_latest_checkpoint_dir(self.estimators_dir)
            logging.info('Loading model from {}'.format(predictor_loc))
            return make_predictions.load_predictor(predictor_loc)


    def save_metrics(self, subjects, samples, acquisitions):
        metrics = check_uncertainty.Model(samples, )