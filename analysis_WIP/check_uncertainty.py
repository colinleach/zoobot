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
from zoobot.uncertainty import dropout_calibration
from zoobot.estimators import input_utils
from zoobot.tfrecord import catalog_to_tfrecord


def predict_input_func(tfrecord_loc, n_galaxies=128):
    """Wrapper to mimic the run_estimator.py input procedure.
    Get subjects and labels from tfrecord, just like during training

    Args:
        tfrecord_loc (str): tfrecord to read subjects from. Should be test data.
        n_galaxies (int, optional): Defaults to 128. Num of galaxies to predict on, as single batch.
    
    Returns:
        subjects: np.array of shape (batch, x, y, channel)
        labels: np.array of shape (batch)
    """

    with tf.Session() as sess:
        config = input_utils.InputConfig(
            name='predict',
            tfrecord_loc=tfrecord_loc,
            label_col='label',
            stratify=False,
            shuffle=False,
            repeat=False,
            stratify_probs=None,
            regression=True,
            geometric_augmentation=False,
            photographic_augmentation=False,
            max_zoom=1.2,
            fill_mode='wrap',
            batch_size=n_galaxies,
            initial_size=128,
            final_size=64,
            channels=3,
            noisy_labels=False  # important - we want the actual vote fractions
        )
        subjects, labels = input_utils.load_batches(config)
        subjects, labels = sess.run([subjects, labels])
    return subjects, labels


def default_metrics(results, labels):
    mean_prediction = results.mean(axis=1)
    
    abs_error = np.abs(results.mean(axis=1) - labels)
    square_error = (labels - mean_prediction) ** 2.
    mean_abs_error = np.mean(abs_error)
    mean_square_error = np.mean(square_error)

    bin_likelihood = make_predictions.binomial_likelihood(labels, results, total_votes=40)
    bin_loss_per_sample = - bin_likelihood  # we want to minimise the loss to maximise the likelihood
    bin_loss_per_subject = np.mean(bin_loss_per_sample, axis=1)
    mean_bin_loss = np.mean(bin_loss_per_subject)  # scalar, mean likelihood

    return mean_prediction, abs_error, square_error, mean_abs_error, mean_square_error, bin_likelihood, bin_loss_per_subject, mean_bin_loss


def save_metrics(results, subjects, labels, save_dir):
    """Describe the performance of prediction results with paper-quality figures.
    
    Args:
        results (np.array): predictions of shape (galaxy_n, sample_n)
        subjects (np.array): galaxies on which predictions were made, shape (batch, x, y, channel)
        labels (np.array): true labels for galaxies on which predictions were made
        save_dir (str): directory into which to save figures of metrics
    """
    mean_prediction, abs_error, square_error, mean_abs_error, mean_square_error, bin_likelihood, bin_loss_per_subject, mean_bin_loss = default_metrics(results, labels)

    print(bin_loss_per_subject)
    # repeat for baseline
    # baseline_results = np.ones_like(results) * labels.mean()  # sample always predicts the mean label
    baseline_results = np.loadtxt('/Data/repos/zoobot/analysis_WIP/uncertainty/al-binomial/five_conv_mse/results.txt')  # baseline is the same model with deterministic labels and MSE loss
    _, baseline_abs_error, baseline_square_error, baseline_mean_abs_error, baseline_mean_square_error, baseline_bin_likelihood, baseline_bin_loss_per_subject, baseline_mean_bin_loss = default_metrics(baseline_results, labels)

    logging.info('Mean square error: {}'.format(mean_square_error))
    logging.info('Baseline mean square error: {}'.format(baseline_mean_square_error))
    logging.info('Mean absolute error: {}'.format(mean_abs_error))
    logging.info('Baseline mean absolute error: {}'.format(baseline_mean_abs_error))
    logging.info('Mean binomial loss: {}'.format(mean_bin_loss))
    logging.info('Baseline mean binomial loss: {}'.format(baseline_mean_bin_loss))

    distribution_entropy = make_predictions.distribution_entropy(results)

    predictive_entropy = make_predictions.predictive_binary_entropy(results)
    expected_entropy = make_predictions.mean_binomial_entropy(results)  # mean over samples
    mutual_info = predictive_entropy - expected_entropy

    
    sns.set(context='paper', font_scale=1.5)

    # save histograms of samples, for first 20 galaxies 
    # fig, axes = make_predictions.view_samples(results[:20], labels[:20])
    # fig.tight_layout()
    # axes[-1].set_xlabel('Vote Fraction')
    # fig.savefig(os.path.join(save_dir, 'sample_dist.png'))
    # plt.close(fig)


    # save distribution of binomial

    # save coverage fraction of the model i.e. check calibration of posteriors
    # plt.figure()
    # alpha_eval, coverage_at_alpha = dropout_calibration.check_coverage_fractions(results, labels)
    # save_loc = os.path.join(save_dir, 'model_coverage.png')
    # dropout_calibration.visualise_calibration(alpha_eval, coverage_at_alpha, save_loc)
    # plt.close()

    # save distribution of various error, compared against baseline that predicts the mean
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))

    ax0.hist(abs_error, label='Model', density=True, alpha=0.5)
    ax0.hist(baseline_abs_error, label='Baseline', density=True, alpha=0.5)
    ax0.set_xlabel('Absolute Error')

    ax1.hist(square_error, label='Model', density=True, alpha=0.5)
    ax1.hist(baseline_square_error, label='Baseline', density=True, alpha=0.5)
    ax1.set_xlabel('Square Error')

    ax2.hist(bin_loss_per_subject, label='Model', density=True, alpha=0.5)
    ax2.hist(baseline_bin_loss_per_subject, label='Baseline', density=True, alpha=0.5)
    ax2.set_xlabel('Binomial Error')

    fig.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics_vs_baseline.png'))
    plt.close()


    # Binomial loss should increase with absolute error, but not linearly
    plt.figure()
    g = sns.jointplot(abs_error, bin_loss_per_subject, kind='reg')
    plt.xlabel('Abs. Error')
    plt.ylabel('Binomial Loss')
    plt.xlim([0., 0.5])
    plt.tight_layout()
    g.savefig(os.path.join(save_dir, 'bin_loss_vs_abs_error.png'))
    plt.close()

    
    # How does being smooth or featured affect each entropy measuremement?
    # Expect expected entropy to be only func. of mean prediction
    # And currently, similarly for predictive entropy (although queried with Lewis)
    fig, (row0, row1) = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12, 8))

    ax00, ax01, ax02 = row0
    ax00.scatter(labels, predictive_entropy)
    ax00.set_ylabel('Predictive Entropy')
    ax01.scatter(labels, expected_entropy)
    ax01.set_ylabel('Expected Entropy')
    ax02.scatter(labels, mutual_info)
    ax02.set_ylabel('Mutual Information')

    ax10, ax11, ax12 = row1
    ax10.scatter(mean_prediction, predictive_entropy)
    ax10.set_ylabel('Predictive Entropy')
    ax11.scatter(mean_prediction, expected_entropy)
    ax11.set_ylabel('Expected Entropy')
    ax12.scatter(mean_prediction, mutual_info)
    ax12.set_ylabel('Mutual Information')

    ax00.set_xlabel('Vote Fraction')
    ax01.set_xlabel('Vote Fraction')
    ax02.set_xlabel('Vote Fraction')
    ax10.set_xlabel('Mean Prediction')
    ax11.set_xlabel('Mean Prediction')
    ax12.set_xlabel('Mean Prediction')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'entropy_by_label.png'))
    plt.close()

    
    # plt.figure()
    # g = sns.jointplot(predictive_entropy, bin_loss, kind='reg')
    # plt.xlabel('Predictive Entropy')
    # plt.ylabel('Binomial Loss')
    # # plt.ylim([0., 0.5])
    # fig.tight_layout()
    # g.savefig(os.path.join(save_dir, 'pred_entropy_vs_bin_loss.png'))
    # plt.close()

    # save correlation between entropy and error. Entropy is more often used for classification.
    # plt.figure()
    # g = sns.jointplot(mutual_info, bin_loss, kind='reg')
    # plt.xlabel('Mutual Information')
    # plt.ylabel('Binomial Loss')
    # # plt.ylim([0., 0.5])
    # fig.tight_layout()
    # g.savefig(os.path.join(save_dir, 'mutual_info_vs_bin_loss.png'))
    # plt.close()

    # save correlation between entropy and error. Entropy is more often used for classification.
    # plt.figure()
    # g = sns.jointplot(mutual_info, abs_error, kind='reg')
    # plt.xlabel('Mutual Information')
    # plt.ylabel('Absolute Error')
    # plt.ylim([0., 0.5])
    # fig.tight_layout()
    # g.savefig(os.path.join(save_dir, 'mutual_info_vs_abs_error.png'))
    # plt.close()

    # save correlation between sample variance and error. Variance is often used for regression.
    # variance is only meaningful for unimodal distributions, keep an eye on this!
    # plt.figure()
    # variance = np.log10(make_predictions.sample_variance(results))
    # g = sns.jointplot(variance, np.abs(results.mean(axis=1) - labels), kind='reg')
    # plt.xlabel('Log Sample Variance')
    # plt.ylabel('Abs. Error')
    # # plt.xlim([-3.5, -2.])
    # plt.ylim([0., 0.5])
    # fig.tight_layout()
    # g.savefig(os.path.join(save_dir, 'variance_correlation.png'))
    # plt.close()

    # show galaxies with max/min variance, or top/bottom 20% of variance (more representative)
    # min_var_gals = subjects[variance.argsort()]
    # max_var_gals = subjects[variance.argsort()[-1::-1]]
    # low_var_galaxies = subjects[variance.argsort()[:int(len(subjects)/5.)]]
    # high_var_galaxies = subjects[variance.argsort()[int(len(subjects)*4./5.):]]
    # np.random.shuffle(low_var_galaxies)   # inplace
    # np.random.shuffle(high_var_galaxies)  # inplace

    # galaxies_to_show = [
    #     {
    #         'galaxies': min_var_gals, 
    #         'save_loc': os.path.join(save_dir, 'min_variance.png')
    #     },
    #     {
    #         'galaxies': max_var_gals,
    #         'save_loc': os.path.join(save_dir, 'max_variance.png')
    #     },
    #     {
    #         'galaxies': high_var_galaxies,
    #         'save_loc': os.path.join(save_dir, 'high_variance.png')
    #     },
    #     {
    #         'galaxies': low_var_galaxies,
    #         'save_loc': os.path.join(save_dir, 'low_variance.png')
    #     },
    # ]

    # save images
    # TODO refactor into astro_utils
    # for galaxy_set in galaxies_to_show:
    #     assert len(galaxy_set['galaxies']) != 0
    #     grid_height = 9
    #     grid_width = 3
    #     fig = plt.figure(figsize=(grid_width * 4, grid_height * 4))  # x, y order
    #     gs1 = gridspec.GridSpec(grid_height, grid_width, fig)  # rows (y), cols (x) order
    #     gs1.update(wspace=0.025, hspace=0.025)
    #     for n in range(grid_height * grid_width):
    #         ax = plt.subplot(gs1[n])
    #         galaxy = galaxy_set['galaxies'][n, :, :, :]  # n, x, y, channel, in ML style
    #         data = galaxy.squeeze()
    #         ax.imshow(data.astype(np.uint8))
    #         ax.grid(False)
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #         # TODO add labels
    #         # label_str = '{:.2}'.format(label)
    #         # ax.text(60, 110, label_str, fontsize=16, color='r')
    #     # plt.tight_layout()
    #     plt.savefig(galaxy_set['save_loc'], bbox_inches='tight')
    #     plt.close()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # dropouts = ['00', '02', '05', '10', '50', '90', '95']
    # dropouts=['05']

    # for dropout in dropouts:

    predictor_names = ['five_conv_mse', 'five_conv_noisy']

    for predictor_name in predictor_names:

        predictor_loc = os.path.join('/data/repos/zoobot/results', predictor_name)

        # predictor_loc = '/Data/repos/zoobot/runs/bayesian_panoptes_featured_si128_sf64_lfloat_no_pred_dropout/final_d{}'.format(dropout)
        model = make_predictions.load_predictor(predictor_loc)

        size = 128
        channels = 3
        feature_spec = read_tfrecord.matrix_label_feature_spec(size=size, channels=channels, float_label=True)

        tfrecord_loc = '/data/repos/zoobot/data/panoptes_featured_s128_lfloat_test.tfrecord'
        subjects, labels = predict_input_func(tfrecord_loc, n_galaxies=1024)

        # save_dir = 'analysis_WIP/uncertainty/dropout_{}'.format(dropout)
        save_dir = 'analysis_WIP/uncertainty/al-binomial/{}'.format(predictor_name)
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
