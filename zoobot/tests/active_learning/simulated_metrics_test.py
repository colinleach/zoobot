import pytest

import os

import pandas as pd

from zoobot.active_learning import simulated_metrics


@pytest.fixture()
def catalog(): 
    # overrides conftest
    return pd.DataFrame([
        {'subject_id': '12'},
        {'subject_id': '57'},
        {'subject_id': '365'}
    ])


@pytest.fixture()
def id_strs():
    return ['12', '57']


def test_match_id_strs_to_catalog(id_strs, catalog):
    # tfrecord from conftest has id_str {'0', ..., '127'}
    df = simulated_metrics.match_id_strs_to_catalog(id_strs, catalog)
    assert sorted(list(df['subject_id'])) == ['12', '57']


# TODO model needs a fixture
def test_show_coverage(model, save_dir):
    model.show_coverage(save_dir)
    assert os.path.exists(os.path.join(save_dir, 'discrete_coverage.png'))
