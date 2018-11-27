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


def entropy(probabilites):
    """Find the total entropy in a sampled probability distribution

    Args:
        probabilites (np.array): observed probabilities e.g. calibrated class scores from ML model
    
    Returns:
        float: total entropy in distribution
    """
    # do p * log p for every sample, sum for each subject
    probabilites = np.clip(probabilites, 0., 1.)
    return -np.sum(list(map(lambda p: p * np.log(p + 1e-12), probabilites)), axis=1)


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
        values_array = entropy(samples)  # calculate on ndarray for speed
        return [float(values_array[n]) for n in range(len(values_array))]  # return a list
    return acquisition_callable


def view_samples(scores, labels, annotate=False):
    """For many subjects, view the distribution of scores and labels for that subject

    Args:
        scores (np.array): class scores, of shape (n_subjects, n_samples)
        labels (np.array): class labels, of shape (n_subjects)
    """
    # correct = (np.mean(scores, axis=1) > 0.5) == labels
    entropies = entropy(scores)  # fast array calculation on all results, look up as needed later

    fig, axes = plt.subplots(len(labels), figsize=(4, len(labels)), sharex=True)
    for galaxy_n, ax in enumerate(axes):
        hist_data = ax.hist(scores[galaxy_n])

        lbound = 0
        ubound = 0.5
        if scores[galaxy_n].mean() > 0.5:
            lbound = 0.5
            ubound = 1

        ax.axvline(labels[galaxy_n], c='k')
        ax.axvline(labels[galaxy_n], c='r')
        # c='r'
        # if correct[galaxy_n]:
        #     c='g'
        # ax.axvspan(lbound, ubound, alpha=0.1, color=c)

        if annotate:
            ax.text(
                0.7, 
                0.75 * np.max(hist_data[0]),
                'H: {:.2}'.format(entropies[galaxy_n])
            )
            ax.set_xlim([0, 1])

    return fig, axes
