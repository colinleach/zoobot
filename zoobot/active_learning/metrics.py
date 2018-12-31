import os
import logging
import json

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


class Model():

    def __init__(self, predictions, labels, name):
        self.predictions = predictions
        self.labels = labels
        self.name = name

        # for speed, calculate the (subject_n, sample_n, k) probabilities once here and re-use
        self.bin_probs = make_predictions.bin_prob_of_samples(predictions, n_draws=40)
        self.mean_prediction = self.predictions.mean(axis=1)

        if self.labels is not None:
            self.calculate_default_metrics()
        self.calculate_acquistion_funcs()


    def calculate_acquistion_funcs(self):
        self.predictive_entropy = acquisition_utils.predictive_binomial_entropy(self.bin_probs)
        self.expected_entropy = np.mean(acquisition_utils.distribution_entropy(self.bin_probs), axis=1)
        self.mutual_info = self.predictive_entropy - self.expected_entropy


    def show_acquisitions_vs_predictions(self, save_dir):
        # How does being smooth or featured affect each entropy measuremement?
        fig, (row0, row1) = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12, 6))
        self.acquisition_vs_mean_prediction(row0)
        self.delta_acquisition_vs_mean_prediction(row1)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'entropy_by_prediction.png'))
        plt.close()


    def acquisition_vs_mean_prediction(self, row):
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

    """
    Evaluating performance given known labels - requires self.labels is not None
    """


    def calculate_default_metrics(self):
        """
        Calculate common metrics for performance of sampled predictions vs. volunteer vote fractions
        Store these in object state
        
        Args:
            results (np.array): predictions of shape (galaxy_n, sample_n)
            labels (np.array): true labels for galaxies on which predictions were made
        
        Returns:
            None
        """
        assert self.labels is not None
        
        self.abs_error = np.abs(self.mean_prediction - self.labels)
        self.square_error = (self.labels - self.mean_prediction) ** 2.
        self.mean_abs_error = np.mean(self.abs_error)
        self.mean_square_error = np.mean(self.square_error)

        self.bin_likelihood = make_predictions.binomial_likelihood(self.labels, self.predictions, total_votes=40)
        self.bin_loss_per_sample = - self.bin_likelihood  # we want to minimise the loss to maximise the likelihood
        self.bin_loss_per_subject = np.mean(self.bin_loss_per_sample, axis=1)
        self.mean_bin_loss = np.mean(self.bin_loss_per_subject)  # scalar, mean likelihood


    def compare_binomial_and_abs_error(self, save_dir):
        # Binomial loss should increase with absolute error, but not linearly
        plt.figure()
        g = sns.jointplot(self.abs_error, self.bin_loss_per_subject, kind='reg')
        plt.xlabel('Abs. Error')
        plt.ylabel('Binomial Loss')
        plt.xlim([0., 0.5])
        plt.tight_layout()
        g.savefig(os.path.join(save_dir, 'bin_loss_vs_abs_error.png'))
        plt.close()


    def show_acquisition_vs_label(self, save_dir):
        fig, row = plt.subplots(ncols=3, sharex=True, figsize=(12, 4))
        _ = self.acquisition_vs_volunteer_votes(row)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'entropy_by_label.png'))
        plt.close()


    def acquisition_vs_volunteer_votes(self, row):
        assert self.labels is not None
        ax00, ax01, ax02 = row
        ax00.scatter(self.labels, self.predictive_entropy)
        ax00.set_ylabel('Predictive Entropy')
        ax01.scatter(self.labels, self.expected_entropy)
        ax01.set_ylabel('Expected Entropy')
        ax02.scatter(self.labels, self.mutual_info)
        ax02.set_ylabel('Mutual Information')

        ax00.set_xlabel('Vote Fraction')
        ax01.set_xlabel('Vote Fraction')
        ax02.set_xlabel('Vote Fraction')

        return ax00, ax01, ax02


    def show_coverage(self, save_dir):
        coverage_df = discrete_coverage.evaluate_discrete_coverage(self.labels, self.bin_probs)
        discrete_coverage.plot_coverage_df(coverage_df, os.path.join(save_dir, 'discrete_coverage.png'))


    def export_performance_metrics(self, save_dir):
        # requires labels. Might be better to extract from the log at execute.py level, via analysis.py.
        data = {}
        data['mean square error'] = self.mean_square_error
        data['mean absolute error'] = self.mean_abs_error
        data['binomial loss'] = self.mean_bin_loss
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(data, f) 
