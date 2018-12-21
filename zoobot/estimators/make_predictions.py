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
    return -np.sum(list(map(lambda p: p * np.log(p + 1e-12), probabilities)), axis=1)


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


def predictive_binary_entropy(probabilities):
    ep = 1e-12
    # eqn 5: mean prediction over MC samples
    mean_prob = np.mean(probabilities, axis=1)
    # eqn 6: usual entropy of that prob
    return -1 * (mean_prob * np.log(mean_prob + ep) + (1 - mean_prob) * np.log(1 - mean_prob + ep))


def mutual_information(probabilities):
    predictive_entropy = binomial_entropy(np.mean(probabilities, axis=1))
    expected_entropy = np.mean(binomial_entropy(probabilities), axis=1)
    return predictive_entropy - expected_entropy


def binomial_entropy(probabilities):
    return np.array(list(map(lambda p:  np.log(p + 1e-12) + np.log(1 - p + 1e-12), probabilities)))


def sample_variance(samples):
    """Mean deviation from the mean. Only meaningful for unimodal distributions.
    See http://mathworld.wolfram.com/SampleVariance.html
    
    Args:
        samples (np.array): predictions of shape (galaxy_n, sample_n)
    
    Returns:
        np.array: variance by galaxy, of shape (galaxy_n)
    """

    return np.apply_along_axis(statistics.variance, arr=samples, axis=1)


def get_acquisition_func(model, n_samples):
    """Provide callable for active learning acquisition
    Args:
        model (function): callable mapping parsed subjects to class scores
        n_samples (int): number of samples (i.e. model calls) to calculate per subject
    
    Returns:
        callable: expects model, returns callable for entropy list of matrix list (given that model)
    """
    def acquisition_callable(subjects):  # subjects must be a list of matrices
        samples = get_samples_of_subjects(model, subjects, n_samples)  # samples is ndarray
        mutual_info = mutual_information(samples) # calculate on ndarray for speed
        return [float(mutual_info[n]) for n in range(len(mutual_info))]  # return a list
    return acquisition_callable


def view_samples(scores, labels, annotate=False):
    """For many subjects, view the distribution of scores and labels for that subject

    Args:
        scores (np.array): class scores, of shape (n_subjects, n_samples)
        labels (np.array): class labels, of shape (n_subjects)
    """
    # correct = (np.mean(scores, axis=1) > 0.5) == labels
    entropies = distribution_entropy(scores)  # fast array calculation on all results, look up as needed later

    fig, axes = plt.subplots(len(labels), figsize=(4, len(labels)), sharex=True, sharey=True)
    for galaxy_n, ax in enumerate(axes):

        x = np.arange(0, 41)
        for score_n, score in enumerate(scores[galaxy_n]):
            if score_n == 0: 
                name = 'Model Posteriors'
            else:
                name = None
            ax.plot(x/40., scipy.stats.binom.pmf(k=x, p=score, n=40), 'k', alpha=0.2, label=name)
        # ax.plot(x/40., scipy.stats.binom.pmf(k=x, p=labels[galaxy_n], n=40), 'r', label='Volunteers')
        ax.axvline(labels[galaxy_n], c='r', label='Observed')
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
