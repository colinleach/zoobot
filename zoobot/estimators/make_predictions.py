import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.contrib import predictor

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn import metrics
import scipy

import seaborn as sns

import statistics  # thanks Python 3.4!

def load_predictor(predictor_loc):
    """Load a saved model as a callable mapping parsed subjects to class scores
    PredictorFromSavedModel expects feeddict of form {examples: batch of data}
    Batch of data should match predict input of estimator, e.g. (?, size, size, channels) shape
    Where '?' is the (flexible) batch dimension

    Args:
        predictor_loc (str): location of predictor (i.e. saved model)

    Returns:
        function: callable expecting parsed subjects according to saved model input configuration
    """
    model_unwrapped = predictor.from_saved_model(predictor_loc)
    # wrap to avoid having to pass around dicts all the time
    # expects image matrix, passes to model within dict of type {examples: matrix}
    # model returns several columns, select 'predictions_for_true' and flip
    # TODO stop flipping, regression problem

    return lambda x: model_unwrapped({'examples': x})['prediction']


def get_samples_of_subjects(model, subjects, n_samples):
    """Get many model predictions on each subject

    Args:
        model (function): callable mapping parsed subjects to class scores
        subjects (list): subject matrices on which to make a prediction
        n_samples (int): number of samples (i.e. model calls) to calculate per subject

    Returns:
        np.array: of form (subject_i, sample_j_of_subject_i)
    """
    results = np.zeros((len(subjects), n_samples))
    for sample_n in range(n_samples):
        results[:, sample_n] = model(subjects)
    return results

    # for nth_run in range(n_samples):  # for each desired sample,
    #     results[:, nth_run] = model(subjects)  # predict once on every example
    # return results


# def get_samples_of_tfrecord(model, tfrecord, n_samples):
#     samples = []
#     for n in range(n_samples):
#         samples.append(model(tfrecord))
#     return np.array(samples)


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


def binomial_likelihood(labels, predictions, total_votes):
    """
    
    In our formalism:
    Labels are v, and labels * total votes are k.
    Predictions are rho.
    Likelihood is minimised (albeit negative) when rho is most likely given k  
    
    Args:
        labels ([type]): [description]
        predictions ([type]): [description]
        total_votes ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    labels = np.expand_dims(labels, 1)
    yes_votes = labels * total_votes
    est_p_yes = np.clip(predictions, 0., 1.)
    epsilon = 1e-8
    return yes_votes * np.log(est_p_yes + epsilon) + (total_votes - yes_votes) * np.log(1. - est_p_yes + epsilon)


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


def bin_prob_of_samples(samples, n_draws):
    # designed to operate on (n_subjects, n_samples) standard response
    assert isinstance(samples, np.ndarray)
    assert len(samples) > 1
    binomial_probs_per_sample = np.zeros(list(samples.shape) + [n_draws + 1])  # add k dimension
    for subject_n in range(samples.shape[0]):
        for sample_n in range(samples.shape[1]):
            rho = samples[subject_n, sample_n]
            binomial_probs_per_sample[subject_n, sample_n, :] = binomial_prob_per_k(rho, n_draws)
    return binomial_probs_per_sample  # (n_subject, n_samples, k)


def binomial_prob_per_k(rho, n_draws):
    """[summary]
    
    Args:
        sampled_rho (float): MLEs of binomial probability, of any dimension
        n_draws (int): N draws for those MLEs.

    Returns:
        (float): entropy of binomial with N draws and p=sampled rho, same shape as inputs
    """
    k = np.arange(0, n_draws + 1)  # include k=n
    return np.array(scipy.stats.binom.pmf(k=k, p=rho, n=n_draws))


def binomial_entropy(rho, n_draws):
    binomial_probs = binomial_prob_per_k(rho, n_draws)
    return distribution_entropy(binomial_probs)
binomial_entropy = np.vectorize(binomial_entropy)


def expected_binomial_entropy(binomial_probs_per_sample):
    # get the entropy over all k (reduce axis 2)
    entropy_per_sample = np.apply_along_axis(distribution_entropy, axis=2, arr=binomial_probs_per_sample)
    return np.mean(entropy_per_sample, axis=1)  # average over samples (reduce axis 1)



def sample_variance(samples):
    """Mean deviation from the mean. Only meaningful for unimodal distributions.
    See http://mathworld.wolfram.com/SampleVariance.html
    
    Args:
        samples (np.array): predictions of shape (galaxy_n, sample_n)
    
    Returns:
        np.array: variance by galaxy, of shape (galaxy_n)
    """

    return np.apply_along_axis(statistics.variance, arr=samples, axis=1)



def mutual_info_acquisition_func(samples):
        bin_probs = bin_prob_of_samples(samples, n_draws=40)  # currently hardcoded
        predictive_entropy = predictive_binomial_entropy(bin_probs)
        expected_entropy = expected_binomial_entropy(bin_probs)
        mutual_info = predictive_entropy - expected_entropy
        return [float(mutual_info[n]) for n in range(len(mutual_info))]  # return a list


def view_samples(scores, labels, annotate=False):
    """For many subjects, view the distribution of scores and labels for that subject

    Args:
        scores (np.array): class scores, of shape (n_subjects, n_samples)
        labels (np.array): class labels, of shape (n_subjects)
    """
    # correct = (np.mean(scores, axis=1) > 0.5) == labels
    # entropies = distribution_entropy(scores)  # fast array calculation on all results, look up as needed later
    x = np.arange(0, 41)

    fig, axes = plt.subplots(len(labels), figsize=(4, len(labels)), sharey=True)

    for galaxy_n, ax in enumerate(axes):
        probability_record = []
        for score_n, score in enumerate(scores[galaxy_n]):
            if score_n == 0: 
                name = 'Model Posteriors'
            else:
                name = None
            probs = binomial_prob_per_k(score, n_draws=40)
            probability_record.append(probs)
            ax.plot(x, probs, 'k', alpha=0.2, label=name)
        probability_record = np.array(probability_record)
        ax.plot(x, probability_record.mean(axis=0), c='g', label='Posterior')
        ax.axvline(labels[galaxy_n] * 40, c='r', label='Observed')
        ax.yaxis.set_visible(False)

    axes[0].legend(
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.1),
        ncol=2, 
        fancybox=True, 
        shadow=False
    )
    fig.tight_layout()

        # hist_data = ax.hist(scores[galaxy_n])

        # lbound = 0
        # ubound = 0.5
        # if scores[galaxy_n].mean() > 0.5:
        #     lbound = 0.5
        #     ubound = 1

        # ax.axvline(labels[galaxy_n], c='k')
        # ax.axvline(labels[galaxy_n], c='r')
        # c='r'
        # if correct[galaxy_n]:
        #     c='g'
        # ax.axvspan(lbound, ubound, alpha=0.1, color=c)

        # if annotate:
        #     ax.text(
        #         0.7, 
        #         0.75 * np.max(hist_data[0]),
        #         'H: {:.2}'.format(entropies[galaxy_n])
        #     )
        #     ax.set_xlim([0, 1])

    return fig, axes
