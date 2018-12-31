import os
import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn import metrics
import tensorflow as tf

from zoobot.estimators import make_predictions, bayesian_estimator_funcs
from zoobot.tfrecord import read_tfrecord
from zoobot.uncertainty import discrete_coverage
from zoobot.estimators import input_utils
from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.active_learning import metrics, acquisition_utils


def compare_model_errors(model_a, model_b, save_dir):
    # save distribution of various error, compared against baseline that predicts the mean
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))

    ax0.hist(model_a.abs_error, label=model_a.name, density=True, alpha=0.5)
    ax0.hist(model_b.abs_error, label=model_b.name, density=True, alpha=0.5)
    ax0.set_xlabel('Absolute Error')

    ax1.hist(model_a.square_error, label=model_a.name, density=True, alpha=0.5)
    ax1.hist(model_b.square_error, label=model_b.name, density=True, alpha=0.5)
    ax1.set_xlabel('Square Error')

    ax2.hist(model_a.bin_loss_per_subject, label=model_a.name, density=True, alpha=0.5)
    ax2.hist(model_b.bin_loss_per_subject, label=model_b.name, density=True, alpha=0.5)
    ax2.set_xlabel('Binomial Error')

    fig.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics_vs_baseline.png'))
    plt.close()


def compare_models(model_a, model_b):
    logging.info('{} mean square error: {}'.format(model_a.name, model_a.mean_square_error))
    logging.info('{} mean square error: {}'.format(model_b.name, model_b.mean_square_error))
    logging.info('{} mean absolute error: {}'.format(model_a.name, model_a.mean_abs_error))
    logging.info('{} mean absolute error: {}'.format(model_b.name, model_b.mean_abs_error))
    logging.info('{} binomial loss: {}'.format(model_a.name, model_a.mean_bin_loss))
    logging.info('{} mean binomial loss: {}'.format(model_b.name, model_b.mean_bin_loss))



def save_metrics(results, subjects, labels, save_dir):
    """Describe the performance of prediction results with paper-quality figures.
    
    Args:
        results (np.array): predictions of shape (galaxy_n, sample_n)
        subjects (np.array): galaxies on which predictions were made, shape (batch, x, y, channel)
        labels (np.array): true labels for galaxies on which predictions were made
        save_dir (str): directory into which to save figures of metrics
    """

    sns.set(context='paper', font_scale=1.5)
    # save histograms of samples, for first 20 galaxies 
    fig, axes = make_predictions.view_samples(results[:20], labels[:20])
    fig.tight_layout()
    axes[-1].set_xlabel(r'Volunteer Vote Fraction $\frac{k}{N}$')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'sample_dist.png'))
    plt.close(fig)

    binomial_metrics = metrics.Model(results, labels, name='binomial')

    # repeat for baseline
    baseline_results = np.ones_like(results) * labels.mean()  # sample always predicts the mean label
    baseline_metrics = metrics.Model(baseline_results, labels, name='baseline')
    # warning: fixed to disk location of this reference model
    mse_results = np.loadtxt('/Data/repos/zoobot/analysis/uncertainty/al-binomial/five_conv_mse/results.txt')  # baseline is the same model with deterministic labels and MSE loss
    mse_metrics = metrics.Model(mse_results, labels, name='mean_loss')

    compare_models(binomial_metrics, baseline_metrics)

    compare_models(binomial_metrics, mse_metrics)

    compare_model_errors(binomial_metrics, mse_metrics, save_dir)
    binomial_metrics.compare_binomial_and_abs_error(save_dir)
    binomial_metrics.show_acquisition_vs_label(save_dir)

    acquisition_utils.save_acquisition_examples(subjects, binomial_metrics.mutual_info, 'mutual_info', save_dir)



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # dropouts = ['00', '02', '05', '10', '50', '90', '95']
    # dropouts=['05']

    # for dropout in dropouts:

    # predictor_names = ['five_conv_noisy']
    predictor_names = ['five_conv_fractions']
    # predictor_names = ['five_conv_mse', 'five_conv_noisy']
    # predictor_names = ['c2548d0_d90']

    for predictor_name in predictor_names:

        predictor_loc = os.path.join('/data/repos/zoobot/results', predictor_name)

        # predictor_loc = '/Data/repos/zoobot/runs/bayesian_panoptes_featured_si128_sf64_lfloat_no_pred_dropout/final_d{}'.format(dropout)
        model = make_predictions.load_predictor(predictor_loc)

        size = 128
        channels = 3
        feature_spec = read_tfrecord.matrix_label_feature_spec(size=size, channels=channels, float_label=True)

        tfrecord_loc = '/data/repos/zoobot/data/basic_split/panoptes_featured_s128_lfloat_test.tfrecord'
        subjects_g, labels_g, _ = input_utils.predict_input_func(tfrecord_loc, n_galaxies=1024, initial_size=size, mode='labels')  # tf graph
        with tf.Session() as sess:
            subjects, labels = sess.run([subjects_g, labels_g])

        save_dir = 'analysis/uncertainty/al-binomial/{}'.format(predictor_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        results_loc = os.path.join(save_dir, 'results.txt')

        new_predictions = False
        if new_predictions:
            results = make_predictions.get_samples_of_subjects(model, subjects, n_samples=100)
            np.savetxt(results_loc, results)
        else:
            assert os.path.exists(results_loc)
            results = np.loadtxt(results_loc)

        save_metrics(results, subjects, labels, save_dir)
