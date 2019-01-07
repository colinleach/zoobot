import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from zoobot.estimators import make_predictions
from zoobot.uncertainty import discrete_coverage


class SimulatedModel():
    """
    Calculate and visualise additional metrics (vs. Model) using a provided catalog
    Useful to create more info from a Model, or for internal use within Timeline
    """

    def __init__(self, model, full_catalog):
        self.model = model
        self.catalog = match_id_strs_to_catalog(model.id_strs, full_catalog)

        self.labels = self.catalog['smooth-or-featured_smooth_fraction']
        assert not any(np.isnan(self.labels))
        
        self.calculate_default_metrics()
        self.votes = np.around(self.labels * 40)  # assume 40 votes for everything, for now


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
        
        self.abs_error = np.abs(self.model.mean_prediction - self.labels)
        self.square_error = (self.labels - self.model.mean_prediction) ** 2.
        self.mean_abs_error = np.mean(self.abs_error)
        self.mean_square_error = np.mean(self.square_error)

        self.bin_likelihood = make_predictions.binomial_likelihood(self.labels, self.model.samples, total_votes=40)
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
        ax00, ax01, ax02 = row
        ax00.scatter(self.labels, self.model.predictive_entropy)
        ax00.set_ylabel('Predictive Entropy')
        ax01.scatter(self.labels, self.model.expected_entropy)
        ax01.set_ylabel('Expected Entropy')
        ax02.scatter(self.labels, self.model.mutual_info)
        ax02.set_ylabel('Mutual Information')

        ax00.set_xlabel('Vote Fraction')
        ax01.set_xlabel('Vote Fraction')
        ax02.set_xlabel('Vote Fraction')

        return ax00, ax01, ax02


    def show_coverage(self, save_dir):
        if self.labels is None:
            raise ValueError('Calculating coverage requires volunteer votes to be known')
        fig, ax = plt.subplots()
        coverage_df = discrete_coverage.evaluate_discrete_coverage(self.votes, self.model.bin_probs)
        discrete_coverage.plot_coverage_df(coverage_df, ax=ax)
        fig.tight_layout()
        save_loc = os.path.join(save_dir, 'discrete_coverage.png')
        fig.savefig(save_loc)


    def export_performance_metrics(self, save_dir):
        # requires labels. Might be better to extract from the log at execute.py level, via analysis.py.
        data = {}
        data['mean square error'] = self.mean_square_error
        data['mean absolute error'] = self.mean_abs_error
        data['binomial loss'] = self.mean_bin_loss
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(data, f) 


def match_id_strs_to_catalog(id_strs, catalog):
    filtered_catalog = catalog[catalog['subject_id'].isin(set(id_strs))]
    # id strs is sorted by acquisition - catalog must also become sorted
    # careful - reindexing by int-like strings will actually do int reindexing, not what I want
    filtered_catalog['subject_id'] = filtered_catalog['subject_id'].astype(str)
    sorted_catalog = filtered_catalog.set_index('subject_id', drop=True).reindex(id_strs).reset_index()
    return sorted_catalog