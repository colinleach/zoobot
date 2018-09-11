import json
import ast
import itertools
import os

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')
sns.set()

from zoobot.tfrecord import read_tfrecord
from zoobot.tests import TEST_FIGURE_DIR



def show_subjects_by_iteration(tfrecord_index_loc, n_subjects, size, channels, save_loc):
    with open(tfrecord_index_loc, 'r') as f:
        tfrecord_locs = json.load(f)
        assert isinstance(tfrecord_locs, list)
    
    nrows = len(tfrecord_locs)
    ncols = n_subjects
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows * 3, ncols * 3))

    for iteration_n, tfrecord_loc in enumerate(tfrecord_locs):
        subjects = read_tfrecord.load_examples_from_tfrecord(
            [tfrecord_loc], 
            read_tfrecord.matrix_label_feature_spec(size, channels),
            n_examples=n_subjects)
        # read_tfrecord.show_examples(subjects, size, channels)
        for subject_n, subject in enumerate(subjects):
            read_tfrecord.show_example(subject, size, channels, axes[iteration_n][subject_n])

    fig.tight_layout()
    plt.savefig(save_loc)


def get_metrics_from_log(log_loc):
    with open(log_loc, 'r') as f:
        content = f.readlines()
    log_entries = filter(lambda x: is_eval_log_entry(x) or is_iteration_split(x), content)

    iteration = 0
    eval_data = []
    for entry in log_entries:
        if is_iteration_split(entry):
            iteration += 1
        else:
            entry_data = parse_eval_log_entry(entry)
            entry_data['iteration'] = iteration
            eval_data.append(entry_data)
        
    metrics = pd.DataFrame(list(eval_data))
    return metrics[metrics['acc/accuracy'] > 0]


def is_iteration_split(line):
    return 'All epochs completed - finishing gracefully' in line


def parse_eval_log_entry(line):
    """[summary]
    
    Args:
        line (str): [description]
    """
    # strip everything left of the colon, and interpret the rest as a literal expression
    colon_index = line.find('global step')
    step_str, data_str = line[colon_index + 12:].split(':')
    step = int(step_str)

    # split on commas, and remove commas
    key_value_strings = data_str.split(',')

    # split on equals, remove equals
    data = {}
    for string in key_value_strings:
        key, value = string.strip(',').split('=')
        data[key.strip()] = float(value.strip())
        data['step'] = step
    return data


def is_eval_log_entry(line):
    return 'Saving dict for global step' in line


def plot_log_metrics(metrics, save_loc):

    # cm = plt.get_cmap('hot') TODO use 'hot' colormap for lines. Really awkward.
    #     cNorm = colors.Normalize(vmin=0, vmax=metrics['iteration'].max())
    #     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    metrics['image_pc'] = ((1 + metrics['iteration']) * 100 / 6).astype(int)
    metrics_list = [metrics[metrics['iteration'] == iteration] for iteration in metrics['iteration'].unique()]

    # smooth (TODO refactor)
    smoothed_list = []
    for df in metrics_list:
        lowess = sm.nonparametric.lowess
        smoothed_metrics = lowess(
            df['acc/accuracy'],
            df['step'],
            is_sorted=True, 
            frac=0.25)
        df['smoothed_acc'] = smoothed_metrics[:, 1]
        smoothed_list.append(df)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))
    for iteration, df in enumerate(smoothed_list):
        ax1.plot(
            df['step'],
            df['smoothed_acc'],
            label='{:3}pc images'.format(iteration * 100 / 5))

        ax2.plot(
            df['step'],
            df['acc/accuracy'],
            label='{:3}pc images'.format(iteration * 100 / 5))

    ax2.set_xlabel('Step')
    ax1.set_ylabel('Eval Accuracy')
    ax2.set_ylabel('Eval Accuracy')
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.savefig(save_loc)


def compare_metrics(all_metrics, save_loc):
    # bars of final acc 
    metrics = pd.concat(all_metrics, axis=0)
    print(metrics.head())
    print(metrics.tail())
    # TODO refactor
    # TODO use 'hue' to add comparison with baseline
    fig, ax = plt.subplots()
    sns.barplot(
        data=metrics[metrics['step'] > 7000], 
        x='image_pc', 
        y='acc/accuracy', 
        hue='name', 
        ax=ax
    )
    ax.set_ylim([0.75, 0.9])
    ax.set_ylabel('Final eval accuracy')
    fig.tight_layout()
    fig.savefig(save_loc)



if __name__ == '__main__':

    active_log_loc = '/Users/mikewalmsley/repos/zoobot/zoobot/logs/execute_1536613916.8920033.log'
    active_save_loc = os.path.join(TEST_FIGURE_DIR, 'active_acc_metrics.png')
    active_metrics = get_metrics_from_log(active_log_loc)
    active_metrics['name'] = 'active'
    plot_log_metrics(active_metrics, active_save_loc)
    
    baseline_log_loc = '/Users/mikewalmsley/repos/zoobot/zoobot/logs/execute_1536613916.8920033.log'
    baseline_save_loc = os.path.join(TEST_FIGURE_DIR, 'baseline_acc_comparison_bar.png')
    baseline_metrics = get_metrics_from_log(baseline_log_loc)
    baseline_metrics['name'] = 'baseline'
    plot_log_metrics(baseline_metrics, baseline_save_loc)

    comparison_save_loc = os.path.join(TEST_FIGURE_DIR, 'acc_bar_comparison.png')
    compare_metrics([baseline_metrics, active_metrics], comparison_save_loc)