import os
import logging
import json
import pickle
from collections import namedtuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn import metrics
import tensorflow as tf

from zoobot.estimators import make_predictions, bayesian_estimator_funcs, input_utils
from zoobot.tfrecord import read_tfrecord
from zoobot.uncertainty import discrete_coverage
from zoobot.active_learning import acquisition_utils
from zoobot.tfrecord import catalog_to_tfrecord

"""Useful for basic recording from iterations. Input to Model, should not be used elsewhere"""
IterationState = namedtuple('IterationState', ['samples', 'acquisitions', 'id_strs'])


def save_iteration_state(iteration_dir, subjects, samples, acquisitions):
    id_strs = [subject['id_str'] for subject in subjects]
    iteration_state = IterationState(samples, acquisitions, id_strs)
    with open(os.path.join(iteration_dir, 'state.pickle'), 'wb') as f:
        pickle.dump(iteration_state, f)


def load_iteration_state(iteration_dir):
    with open(os.path.join(iteration_dir, 'state.pickle'), 'rb') as f:
        return pickle.load(f)



class Model():
    """Get and plot basic model results, with no external info"""

    def __init__(self, state, name, bin_probs=None):
        # save sorted by acq. value (descending), to avoid resorting later
        args_to_sort = np.argsort(state.acquisitions)[::-1]
        self.samples = state.samples[args_to_sort, :]
        self.id_strs = [state.id_strs[n] for n in args_to_sort]  # list, sort with listcomp
        self.acquisitions = [state.acquisitions[n] for n in args_to_sort]
        self.name = name

        # for speed, calculate the (subject_n, sample_n, k) probabilities once here and re-use
        if bin_probs is None:
            self.bin_probs = make_predictions.bin_prob_of_samples(self.samples, n_draws=40)
        else:
            self.bin_probs = bin_probs
        
        self.mean_prediction = self.samples.mean(axis=1)
        self.calculate_mutual_info()
        

    def calculate_mutual_info(self):
        self.predictive_entropy = acquisition_utils.predictive_binomial_entropy(self.bin_probs)
        self.expected_entropy = np.mean(acquisition_utils.distribution_entropy(self.bin_probs), axis=1)
        self.mutual_info = self.predictive_entropy - self.expected_entropy


    def show_mutual_info_vs_predictions(self, save_dir):
        # How does being smooth or featured affect each entropy measuremement?
        fig, (row0, row1) = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12, 6))
        self.mutual_info_vs_mean_prediction(row0)
        self.delta_acquisition_vs_mean_prediction(row1)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'entropy_by_prediction.png'))
        plt.close()


    def mutual_info_vs_mean_prediction(self, row):
        mean_pred_range = np.linspace(0.02, 0.98)
        entropy_of_mean_pred_range = acquisition_utils.binomial_entropy(mean_pred_range, n_draws=40)

        row[0].scatter(self.mean_prediction, self.predictive_entropy)
        row[0].plot(mean_pred_range, entropy_of_mean_pred_range)
        row[0].set_xlabel('Mean Prediction')
        row[0].set_ylabel('Predictive Entropy')

        row[1].scatter(self.mean_prediction, self.expected_entropy)
        row[1].plot(mean_pred_range, entropy_of_mean_pred_range)
        row[1].set_ylabel('Expected Entropy')
        row[2].scatter(self.mean_prediction, self.mutual_info)
        row[2].set_ylabel('Mutual Information')


        row[1].set_xlabel('Mean Prediction')
        row[2].set_xlabel('Mean Prediction')


    def delta_acquisition_vs_mean_prediction(self, row):
        entropy_of_mean_prediction = acquisition_utils.binomial_entropy(self.mean_prediction, n_draws=40)
        row[0].scatter(self.mean_prediction, self.predictive_entropy - entropy_of_mean_prediction)
        row[0].set_ylabel('Delta Predictive Entropy')
        row[1].scatter(self.mean_prediction, self.expected_entropy - entropy_of_mean_prediction)
        row[1].set_ylabel('Delta Expected Entropy')


    def acquisitions_vs_mean_prediction(self, n_acquired, save_dir):
        acquisitions_vs_values(self.acquisitions, self.mean_prediction, n_acquired, 'Mean Prediction', save_dir)


def acquisitions_vs_values(acquisitions, values, n_acquired, xlabel, save_dir):

    verify_ready_to_plot(acquisitions, n_acquired)
    
    acquired, not_acquired = values[:n_acquired], values[n_acquired:]

    fig, axes = plt.subplots(nrows=3, sharex=True)

    sns.scatterplot(
        x=values, 
        y=acquisitions, 
        hue=np.array(acquisitions) > acquisitions[n_acquired],
        ax=axes[0]
        )

    axes[0].set_ylabel('Acquisition')
    axes[0].set_xlabel(xlabel)

    axes[1].hist(acquired)
    axes[1].set_xlabel(xlabel)
    axes[1].set_title('Acquired')

    axes[2].hist(not_acquired)
    axes[2].set_xlabel(xlabel)
    axes[2].set_title('Not Acquired')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'acquistion_vs_{}.png'.format(xlabel.replace(' ', '_').lower())))


def verify_ready_to_plot(acquisitions, n_acquired):
    if acquisitions is None:
        raise ValueError('Acquistions is required')
    if len(acquisitions) < n_acquired:
        raise ValueError('N Acquired is set incorrectly: should be less than all subjects')

    # must already be sorted in descending order (no way to check 'value', ofc)
    assert np.allclose(acquisitions, np.sort(acquisitions)[::-1])
