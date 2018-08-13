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


def load_predictor(predictor_loc):
    """Load a saved model as a callable mapping parsed subjects to class scores

    Args:
        predictor_loc (str): location of predictor (i.e. saved model)

    Returns:
        function: callable expecting parsed subjects according to saved model input configuration
    """
    model_unwrapped = predictor.from_saved_model(predictor_loc)
    # wrap to avoid having to pass around dicts all the time
    return lambda x: 1 - model_unwrapped({'examples': x})['predictions_for_true']


def get_samples_of_subjects(model, subjects, n_samples):
    """Get many model predictions on each subject
    
    Args:
        model (function): callable mapping parsed subjects to class scores
        subjects (list): subjects on which to make a prediction
        n_samples (int): number of samples (i.e. model calls) to calculate per subject
    
    Returns:
        np.array: of form (subject_i, sample_j_of_subject_i)
    """
    results = np.zeros((len(subjects), n_samples))
    
    for nth_run in range(n_samples):  # for each desired sample,
        results[:, nth_run] = model(subjects)  # predict once on every example

    return results


def entropy(probabilites):
    """Find the total entropy in a sampled probability distribution
    
    Args:
        probabilites (np.array): observed probabilities e.g. calibrated class scores from ML model
    
    Returns:
        float: total entropy in distribution
    """
    return -np.sum(list(map(lambda p: p * np.log(p + 1e-12), probabilites)))


def view_samples(scores, labels):
    """For many subjects, view the distribution of scores and labels for that subject
    
    Args:
        scores (np.array): class scores, of shape (n_subjects, n_samples)
        labels (np.array): class labels, of shape (n_subjects)
    """ 
    correct = (np.mean(scores, axis=1) > 0.5) == labels
    fig, axes = plt.subplots(len(labels), figsize=(4, len(labels)), sharex=True)
    for galaxy_n, ax in enumerate(axes):
        hist_data = ax.hist(scores[galaxy_n])
        c='r'
        if correct[galaxy_n]:
            c='g'
        
        lbound = 0
        ubound = 0.5
        if scores[galaxy_n].mean() > 0.5:
            lbound = 0.5
            ubound = 1
            
        ax.axvspan(lbound, ubound, alpha=0.1, color=c)
        ax.text(0.7, 0.75 * np.max(hist_data[0]), 'H: {}'.format(str(entropy(scores[galaxy_n]))[:4]))
        ax.set_xlim([0, 1])
