import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from zoobot.tfrecord import read_tfrecord
from zoobot.active_learning import metrics

class Timeline():
    """
    Create and compare SimulatedModel over many iterations
    """
    def __init__(self, states, catalog, save_dir):
        self.models = simulated_models_over_time(states, catalog)
        self.save_dir = save_dir


    def save_model_histograms(self):
        for attr_str in ['labels', 'ra', 'dec']:
            show_model_attr_hist_by_iteration(self.models, attr_str, self.save_dir)


def read_id_strs_from_tfrecord(tfrecord_loc, max_subjects=1024):
    # useful for verifying the right subjects are in fact saved
    feature_spec = read_tfrecord.id_feature_spec()
    subjects = read_tfrecord.load_examples_from_tfrecord(tfrecord_loc, feature_spec, max_examples=max_subjects)
    id_strs = [subject['id_str'].decode('utf-8') for subject in subjects]
    assert len(set(id_strs)) == len(id_strs)
    return id_strs


def identify_catalog_subjects_history(tfrecord_locs, catalog):
    assert isinstance(tfrecord_locs, list)
    return [metrics.match_id_strs_to_catalog(read_id_strs_from_tfrecord(tfrecord_loc), catalog) for tfrecord_loc in tfrecord_locs]


def show_model_attr_hist_by_iteration(models, attr_str, save_dir):
    fig, axes = plt.subplots(nrows=len(models), sharex=True)
    for iteration_n, model in enumerate(models):
        attr_values = getattr(model, attr_str)
        axes[iteration_n].hist(attr_values, density=True)
    axes[-1].set_xlabel(attr_str)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, attr_str + '_over_time.png'))


def simulated_models_over_time(states, catalog):
    models_by_iter = []
    for state, iteration_n in enumerate(states):
        model = metrics.Model(state.samples, id_strs=state.id_strs, name='iteration_{}'.format(n), acquisitions=state.acquisitions)
        simulated_model = metrics.SimulatedModel(model, catalog)
        models_by_iter.append(model)
    return models_by_iter

