import os
import shutil

import numpy as np


class Iteration():

    def __init__(self, run_dir, iteration_n, initial_estimator_ckpt=None):
        self.iteration_dir = os.path.join(run_dir, 'iteration_{}'.format(iteration_n))
        self.estimators_dir = os.path.join(self.iteration_dir, 'estimators')

        os.mkdir(self.iteration_dir)
        os.mkdir(self.estimators_dir)

        if initial_estimator_ckpt is not None:
            # copy the initial estimator folder inside estimators_dir, keeping the same name
            shutil.copytree(
                src=initial_estimator_ckpt, 
                dst=os.path.join(self.estimators_dir, os.path.split(initial_estimator_ckpt)[-1])
            )


    def make_predictions(self):
        raise NotImplementedError
