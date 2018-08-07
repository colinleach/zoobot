
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


def load_prediction_model(estimator_loc):
    return model


def get_samples_of_examples(model, examples, n_samples):
    results = np.zeros((len(examples), n_samples))
    
    for nth_run in range(n_samples):  # for each desired sample,
        results[:, nth_run] = model(examples)  # predict once on every example

    return results


def entropy(p_values):
    return -np.sum(list(map(lambda p: p * np.log(p + 1e-12), p_values)))


def view_samples(scores, labels, selected=None):
    
    if selected is not None:
        scores = scores[selected]
        labels = labels[selected]

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
#         ax.text(0.1, 400, '{}'.format(galaxy_n))
        ax.set_xlim([0, 1])

if __name__ == '__main__':

n_examples = 200  # number of train/test examples to make predictions on
n_samples = 50  # predict this many times per example
model_train_results = [get_samples_of_examples(model, train_examples[:n_examples], n_samples) for model in models]
model_test_results = [get_samples_of_examples(model, test_examples[:n_examples], n_samples) for model in models]

sns.set_context('notebook')
PACKAGE_DIR = '/Data/repos/zoobot/zoobot'

untrained_model_loc = PACKAGE_DIR + '/runs/bayesian_panoptes_featured_si128_sf64_l0.4_augs_both_normed_activated_wide/1530201507'  # needs update
midtrained_model_loc = PACKAGE_DIR + '/runs/bayesian_panoptes_featured_si128_sf64_l0.4_augs_both_normed_activated_wide/1530242652' 
trained_model_loc = PACKAGE_DIR + '/runs/bayesian_panoptes_featured_si128_sf64_l0.4_augs_both_normed_activated_wide/1530286779' 


untrained_model_unwrapped = predictor.from_saved_model(untrained_model_loc)
midtrained_model_unwrapped = predictor.from_saved_model(midtrained_model_loc)
trained_model_unwrapped = predictor.from_saved_model(trained_model_loc)

# wrap to avoid having to pass around dicts all the time
untrained_model = lambda x: 1 - untrained_model_unwrapped({'examples': x})['predictions_for_true']
midtrained_model = lambda x: 1 - midtrained_model_unwrapped({'examples': x})['predictions_for_true']
trained_model = lambda x: 1 - trained_model_unwrapped({'examples': x})['predictions_for_true']

models = [untrained_model, midtrained_model, trained_model]