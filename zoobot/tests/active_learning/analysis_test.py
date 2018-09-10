import os
import json

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zoobot.tests import TEST_FIGURE_DIR, TEST_EXAMPLE_DIR
from zoobot.active_learning import analysis


@pytest.fixture()
def tfrecord_index_loc(tfrecord_dir, example_tfrecord_loc):
    loc = os.path.join(tfrecord_dir, 'tfrecord_index.json')
    with open(loc, 'w') as f:
        json.dump([example_tfrecord_loc, example_tfrecord_loc], f)
    return loc

def test_show_subjects_by_iteration(tfrecord_index_loc, size, channels):
    save_loc = os.path.join(TEST_FIGURE_DIR, 'subjects_in_shards.png')
    analysis.show_subjects_by_iteration(tfrecord_index_loc, 5, size, channels, save_loc)


# TODO temporary, will need one in TEST_EXAMPLES
@pytest.fixture()
def log_loc():
    return '/Users/mikewalmsley/repos/zoobot/zoobot/logs/bayesian_panoptes_featured_si128_sf64_l0.4_augs_both_normed_activated_wide_aws.log'


@pytest.fixture()
def eval_log_entry():
    return '2018-09-10 12:03:37,451 Saving dict for global step 841: acc/accuracy = 0.7994792, acc/mean_per_class_accuracy = 0.79981816, confusion/false_negatives = 35.0, confusion/false_positives = 42.0, confusion/true_negatives = 155.0, confusion/true_positives = 152.0, global_step = 841, loss = 0.40684244, pr/auc = 0.79981816, pr/precision = 0.78350514, pr/recall = 0.8128342, sanity/predictions_above_95% = 0.1015625, sanity/predictions_below_5% = 0.13802083'


def test_is_eval_log_entry(eval_log_entry):
    assert analysis.is_eval_log_entry(eval_log_entry)


def test_parse_eval_log_entry(eval_log_entry):
    data = analysis.parse_eval_log_entry(eval_log_entry)
    assert np.isclose(data['acc/accuracy'], 0.7994792)
    assert np.isclose(data['sanity/predictions_below_5%'], 0.13802083)


def test_get_metrics_from_log(log_loc):
    metrics = analysis.get_metrics_from_log(log_loc)
    metrics.to_csv(os.path.join(TEST_EXAMPLE_DIR, 'test_metrics.csv'))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(metrics['acc/accuracy'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Eval Accuracy')
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'baseline_acc.png'))