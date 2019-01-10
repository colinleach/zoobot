import os
import statistics  # thanks Python 3.4!

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from shared_astro_utils import plotting_utils
from zoobot.estimators import make_predictions


def distribution_entropy(probabilities):
    """Find the total entropy in a sampled probability distribution
    Done accidentally - should really by summing over each class, according to Lewis
    However, highly useful!

    Args:
        probabilites (np.array): observed probabilities e.g. calibrated class scores from ML model
    
    Returns:
        float: total entropy in distribution
    """
    # do p * log p for every sample, sum for each subject
    probabilities = np.clip(probabilities, 0., 1.)
    return -np.sum(list(map(lambda p: p * np.log(p + 1e-12), probabilities)), axis=-1)



def predictive_binomial_entropy(binomial_probs_per_sample):
    """[summary]
    
    Args:
        sampled_rho (float): MLEs of binomial probability, of any dimension
        n_draws (int): N draws for those MLEs.
    
    Returns:
        (float): entropy of binomial with N draws and p=sampled rho, same shape as inputs
    """
    # average over samples to get the mean prediction per k, per subject
    bin_probs_per_k_per_subject = np.mean(binomial_probs_per_sample, axis=1)
    return distribution_entropy(bin_probs_per_k_per_subject)


def binomial_entropy(rho, n_draws):
    """
    If possible, calculate bin probs only once, for speed
    Only use this function when rho is only used here
    """
    binomial_probs = make_predictions.binomial_prob_per_k(rho, n_draws)
    return distribution_entropy(binomial_probs)
binomial_entropy = np.vectorize(binomial_entropy)


def expected_binomial_entropy(binomial_probs_per_sample):
    # get the entropy over all k (reduce axis 2)
    entropy_per_sample = np.apply_along_axis(distribution_entropy, axis=2, arr=binomial_probs_per_sample)
    return np.mean(entropy_per_sample, axis=1)  # average over samples (reduce axis 1)


def mutual_info_acquisition_func(samples):
        bin_probs = make_predictions.bin_prob_of_samples(samples, n_draws=40)  # currently hardcoded
        predictive_entropy = predictive_binomial_entropy(bin_probs)
        expected_entropy = expected_binomial_entropy(bin_probs)
        mutual_info = predictive_entropy - expected_entropy
        return [float(mutual_info[n]) for n in range(len(mutual_info))]  # return a list



def sample_variance(samples):
    """Mean deviation from the mean. Only meaningful for unimodal distributions.
    See http://mathworld.wolfram.com/SampleVariance.html
    
    Args:
        samples (np.array): predictions of shape (galaxy_n, sample_n)
    
    Returns:
        np.array: variance by galaxy, of shape (galaxy_n)
    """

    return np.apply_along_axis(statistics.variance, arr=samples, axis=1)


def save_acquisition_examples(tfrecord_locs, id_strs, acq_values, acq_string, save_dir):
    """[summary]
    
    Args:
        subject_data (np.array): of form [n_subjects, height, width, channels]. NOT a list.
        acq_values ([type]): [description]
        acq_string ([type]): [description]
        save_dir ([type]): [description]
    """
    assert isinstance(tfrecord_loc, str)
    assert isinstance(acq_values, np.ndarray)
    # show galaxies with max/min variance, or top/bottom 20% of variance (more representative)
    sorted_galaxies = subject_data[acq_values.argsort()]
    min_gals = sorted_galaxies
    max_gals = sorted_galaxies[::-1]  # reverse
    low_galaxies = sorted_galaxies[:int(len(subject_data)/5.)]
    high_galaxies = sorted_galaxies[int(-len(subject_data)/5.):]
    np.random.shuffle(low_galaxies)   # inplace
    np.random.shuffle(high_galaxies)  # inplace

    galaxies_to_show = [
        {
            'galaxies': min_gals, 
            'save_loc': os.path.join(save_dir, 'min_{}.png'.format(acq_string))
        },
        {
            'galaxies': max_gals,
            'save_loc': os.path.join(save_dir, 'max_{}.png'.format(acq_string))
        },
        {
            'galaxies': high_galaxies,
            'save_loc': os.path.join(save_dir, 'high_{}.png'.format(acq_string))
        },
        {
            'galaxies': low_galaxies,
            'save_loc': os.path.join(save_dir, 'low_{}.png'.format(acq_string))
        },
    ]

    # save images
    for galaxy_set in galaxies_to_show:
        assert len(galaxy_set['galaxies']) != 0
        plotting_utils.plot_galaxy_grid(
            galaxies=galaxy_set['galaxies'],
            rows=9,
            columns=3,
            save_loc=galaxy_set['save_loc']
        )
