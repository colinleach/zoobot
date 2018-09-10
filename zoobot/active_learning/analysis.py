import json
import ast

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zoobot.tfrecord import read_tfrecord


def get_metrics_from_log(log_loc):
    with open(log_loc, 'r') as f:
        content = f.readlines()
    eval_log_entries = filter(is_eval_log_entry, content)
    eval_data = map(parse_eval_log_entry, eval_log_entries)
    return pd.DataFrame(list(eval_data))



def parse_eval_log_entry(line):
    """[summary]
    
    Args:
        line (str): [description]
    """
    # strip everything left of the colon, and interpret the rest as a literal expression
    colon_index = line.find(': ')
    data_str = line[colon_index + 1:]

    # split on commas, and remove commas
    key_value_strings = data_str.split(',')

    # split on equals, remove equals
    data = {}
    for string in key_value_strings:
        key, value = string.strip(',').split('=')
        data[key.strip()] = float(value.strip())
    return data


def is_eval_log_entry(line):
    return 'Saving dict for global step' in line


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

