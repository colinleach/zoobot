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


def save_metrics(results, subjects, labels, save_dir):
    """Describe the performance of prediction results with paper-quality figures.
    
    Args:
        results (np.array): predictions of shape (galaxy_n, sample_n)
        subjects (np.array): galaxies on which predictions were made, shape (batch, x, y, channel)
        labels (np.array): true labels for galaxies on which predictions were made
        save_dir (str): directory into which to save figures of metrics
    """


    sns.set(context='paper', font_scale=1.5)

    mse = metrics.mean_squared_error(labels, results.mean(axis=1))
    logging.info('Mean mse: {}'.format(mse))

    baseline_mse = metrics.mean_squared_error(labels, np.ones_like(labels) * labels.mean())
    logging.info('Baseline mse: {}'.format(baseline_mse))

    # save histograms of samples, for first 20 galaxies 
    fig, axes = make_predictions.view_samples(results[:20], labels[:20])
    fig.tight_layout()
    axes[-1].set_xlabel('Vote Fraction')
    fig.savefig(os.path.join(save_dir, 'sample_dist.png'))
    plt.close(fig)

    # save distribution of mean squared error, compared against baseline that predicts the mean
    plt.figure()
    plt.hist(np.abs(results.mean(axis=1) - labels), label='Model', density=True, alpha=0.5)
    plt.hist(np.abs(labels.mean() - labels), label='Baseline', density=True, alpha=0.5)
    plt.legend()
    plt.xlabel('Mean Square Error')
    plt.savefig(os.path.join(save_dir, 'mean_square_error_dist.png'))
    plt.close()


    # save coverage fraction of the model i.e. check calibration of posteriors
    plt.figure()
    alpha_eval, coverage_at_alpha = dropout_calibration.check_coverage_fractions(results, labels)
    save_loc = os.path.join(save_dir, 'model_coverage.png')
    dropout_calibration.visualise_calibration(alpha_eval, coverage_at_alpha, save_loc)
    plt.close()


    # tf graph for binomial loss via my custom func
    # labels_p = tf.placeholder()
    # predictions_p = tf.placeholder()
    # binomial_loss = bayesian_estimator_funcs.binomial_loss(labels_p, predictions_p)

    # with tf.Session() as sess:
    #     bin_loss = sess.run([binomial_loss], feeddict={'labels_p': labels, 'predictions_p': })

    abs_error = np.abs(results.mean(axis=1) - labels)
    # TODO refactor into function
    p_yes = np.mean(results, axis=1)  # typical prediction for typical loss: mean over samples
    # could alternatively do loss for each sample, averaged?
    total_votes = 40
    yes_votes = p_yes * total_votes
    epsilon = 1e-6
    bin_loss = - (yes_votes * np.log(p_yes + epsilon) + (total_votes - yes_votes) * np.log(1 - p_yes + epsilon))

    entropy = make_predictions.entropy(results)
    expected_entropy = make_predictions.mean_binomial_entropy(results)
    mutual_info = entropy - expected_entropy


    # save correlation between entropy and error. Entropy is more often used for classification.
    plt.figure()
    g = sns.jointplot(bin_loss, abs_error, kind='reg')
    plt.xlabel('Binomial Loss')
    plt.ylabel('Abs. Error')
    plt.ylim([0., 0.5])
    fig.tight_layout()
    g.savefig(os.path.join(save_dir, 'bin_loss_vs_abs_error.png'))
    plt.close()


    # save correlation between entropy and error. Entropy is more often used for classification.
    plt.figure()
    g = sns.jointplot(entropy, bin_loss, kind='reg')
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Binomial Loss')
    # plt.ylim([0., 0.5])
    fig.tight_layout()
    g.savefig(os.path.join(save_dir, 'pred_entropy_vs_bin_loss.png'))
    plt.close()

    # save correlation between entropy and error. Entropy is more often used for classification.
    plt.figure()
    g = sns.jointplot(mutual_info, bin_loss, kind='reg')
    plt.xlabel('Mutual Information')
    plt.ylabel('Binomial Loss')
    # plt.ylim([0., 0.5])
    fig.tight_layout()
    g.savefig(os.path.join(save_dir, 'mutual_info_vs_bin_loss.png'))
    plt.close()

    # save correlation between entropy and error. Entropy is more often used for classification.
    plt.figure()
    g = sns.jointplot(mutual_info, abs_error, kind='reg')
    plt.xlabel('Mutual Information')
    plt.ylabel('Absolute Error')
    plt.ylim([0., 0.5])
    fig.tight_layout()
    g.savefig(os.path.join(save_dir, 'mutual_info_vs_abs_error.png'))
    plt.close()

    # save correlation between sample variance and error. Variance is often used for regression.
    # variance is only meaningful for unimodal distributions, keep an eye on this!
    plt.figure()
    variance = np.log10(make_predictions.sample_variance(results))
    g = sns.jointplot(variance, np.abs(results.mean(axis=1) - labels), kind='reg')
    plt.xlabel('Log Sample Variance')
    plt.ylabel('Abs. Error')
    # plt.xlim([-3.5, -2.])
    plt.ylim([0., 0.5])
    fig.tight_layout()
    g.savefig(os.path.join(save_dir, 'variance_correlation.png'))
    plt.close()

    # show galaxies with max/min variance, or top/bottom 20% of variance (more representative)
    min_var_gals = subjects[variance.argsort()]
    max_var_gals = subjects[variance.argsort()[-1::-1]]
    low_var_galaxies = subjects[variance.argsort()[:int(len(subjects)/5.)]]
    high_var_galaxies = subjects[variance.argsort()[int(len(subjects)*4./5.):]]
    np.random.shuffle(low_var_galaxies)   # inplace
    np.random.shuffle(high_var_galaxies)  # inplace

    galaxies_to_show = [
        {
            'galaxies': min_var_gals, 
            'save_loc': os.path.join(save_dir, 'min_variance.png')
        },
        {
            'galaxies': max_var_gals,
            'save_loc': os.path.join(save_dir, 'max_variance.png')
        },
        {
            'galaxies': high_var_galaxies,
            'save_loc': os.path.join(save_dir, 'high_variance.png')
        },
        {
            'galaxies': low_var_galaxies,
            'save_loc': os.path.join(save_dir, 'low_variance.png')
        },
    ]

    # save images
    # TODO refactor into astro_utils
    for galaxy_set in galaxies_to_show:
        assert len(galaxy_set['galaxies']) != 0
        grid_height = 9
        grid_width = 3
        fig = plt.figure(figsize=(grid_width * 4, grid_height * 4))  # x, y order
        gs1 = gridspec.GridSpec(grid_height, grid_width, fig)  # rows (y), cols (x) order
        gs1.update(wspace=0.025, hspace=0.025)
        for n in range(grid_height * grid_width):
            ax = plt.subplot(gs1[n])
            galaxy = galaxy_set['galaxies'][n, :, :, :]  # n, x, y, channel, in ML style
            data = galaxy.squeeze()
            ax.imshow(data.astype(np.uint8))
            ax.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # TODO add labels
            # label_str = '{:.2}'.format(label)
            # ax.text(60, 110, label_str, fontsize=16, color='r')
        # plt.tight_layout()
        plt.savefig(galaxy_set['save_loc'], bbox_inches='tight')
        plt.close()


if __name__ == '__main__':

    # dropouts = ['00', '02', '05', '10', '50', '90', '95']
    # dropouts=['05']

    # for dropout in dropouts:

    predictor_names = ['five_conv_noisy']


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
