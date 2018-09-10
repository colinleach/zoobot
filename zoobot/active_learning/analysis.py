import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zoobot.tfrecord import read_tfrecord


def show_subjects_by_iteration(tfrecord_index_loc, n_subjects, size, channels, save_loc):
    with open(tfrecord_index_loc, 'r') as f:
        tfrecord_locs = json.load(f)
        assert isinstance(tfrecord_locs, list)
    
    nrows = len(tfrecord_locs)
    ncols = n_subjects
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows * 3, ncols * 3)

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

