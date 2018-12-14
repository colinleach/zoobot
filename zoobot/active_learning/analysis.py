import json
import ast
import itertools
import os
import argparse

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
    return metrics[metrics['rmse'] > 0]


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


def smooth_metrics(metrics_list):
    # smooth (TODO refactor)
    smoothed_list = []
    for df in metrics_list:
        lowess = sm.nonparametric.lowess
        smoothed_metrics = lowess(
            df['rmse'],
            df['step'],
            is_sorted=True, 
            frac=0.25)
        df['smoothed_acc'] = smoothed_metrics[:, 1]
        smoothed_list.append(df)
    return smoothed_list


def plot_log_metrics(metrics_list, save_loc, title=None):

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, sharey=True)
    for df in metrics_list:
        iteration = df['iteration'].iloc[0]
        ax1.plot(
            df['step'],
            df['smoothed_acc'],
            label='Iteration {}'.format(iteration)
        )

        ax2.plot(
            df['step'],
            df['rmse'],
            label='Iteration {}'.format(iteration)
        )

    ax2.set_xlabel('Step')
    ax1.set_ylabel('Eval Accuracy')
    ax2.set_ylabel('Eval Accuracy')
    ax1.legend()
    ax2.legend()
    if title is not None:
        ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(save_loc)


def compare_metrics(all_metrics, save_loc, title=None):
    
    best_rows = []
    for df in all_metrics:
        df = df.reset_index()
        best_acc_idx = df['smoothed_acc'].idxmax()
        best_row = df.iloc[best_acc_idx]
        best_rows.append(best_row)

    metrics = pd.DataFrame(best_rows)


    fig, ax = plt.subplots()
    sns.barplot(
        data=metrics, 
        x='iteration', 
        y='smoothed_acc', 
        hue='name', 
        ax=ax,
        ci=80
    )
    ax.set_ylim([0.75, 0.95])
    ax.set_ylabel('Final eval accuracy')
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_loc)

def split_by_iter(df):
    return [df[df['iteration'] == iteration] for iteration in df['iteration'].unique()]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyse active learning')
    parser.add_argument('--active_dir', dest='active_dir', type=str,
                    help='')
    parser.add_argument('--baseline_dir', dest='baseline_dir', type=str,
                    help='')
    parser.add_argument('--initial', dest='initial', type=int,
                    help='')
    parser.add_argument('--per_iter', dest='per_iter', type=int,
                    help='')
    parser.add_argument('--output_dir', dest='output_dir', type=str,
                    help='')

    args = parser.parse_args()

    initial = args.initial
    per_iter = args.per_iter
    title = 'Initial: {}. Per iter: {}. From scratch.'.format(initial, per_iter)
    name = '{}init_{}per'.format(initial, per_iter)

    active_log_loc = os.path.join(args.active_dir, list(filter(lambda x: '.log' in x, os.listdir(args.active_dir)))[0])  # returns as tuple of (dir, name)
    print(active_log_loc)
    # active_log_loc = os.path.join(args.active_dir, log_name)
    active_save_loc = os.path.join(args.output_dir, 'acc_metrics_active_' + name + '.png')
    active_metrics = get_metrics_from_log(active_log_loc)
    active_metrics['name'] = 'active'
    active_metric_iters = split_by_iter(active_metrics)
    active_metric_smooth = smooth_metrics(active_metric_iters)
    plot_log_metrics(active_metric_smooth, active_save_loc)
    
    # if args.baseline_dir != '':
    #     baseline_log_loc = '/Users/mikewalmsley/repos/zoobot/zoobot/logs/baseline_' + name + '.log'
    #     baseline_save_loc = os.path.join(args.output_dir, 'acc_metrics_baseline_' + name + '.png')
    #     baseline_metrics = get_metrics_from_log(baseline_log_loc)
    #     baseline_metrics['name'] = 'baseline'
    #     baseline_metric_iters = split_by_iter(baseline_metrics)
    #     baseline_metric_smooth = smooth_metrics(baseline_metric_iters)
    #     plot_log_metrics(baseline_metric_smooth, baseline_save_loc, title=title)

    #     comparison_save_loc = os.path.join(TEST_FIGURE_DIR, 'acc_comparison_' + name + '.png')
    #     compare_metrics(baseline_metric_smooth + active_metric_smooth, comparison_save_loc, title=title)
