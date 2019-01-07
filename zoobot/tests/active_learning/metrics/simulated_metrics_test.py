import pytest

import os

import numpy as np
import pandas as pd

from zoobot.active_learning import simulated_metrics


@pytest.fixture()
def tiny_catalog(): 
    # overrides conftest
    return pd.DataFrame([
        {'subject_id': '57'},
        {'subject_id': '365'},
        {'subject_id': '12'}
    ])


@pytest.fixture()
def tiny_id_strs():
    return ['12', '57']


def test_match_id_strs_to_catalog(tiny_id_strs, tiny_catalog):
    df = simulated_metrics.match_id_strs_to_catalog(tiny_id_strs, tiny_catalog)
    assert list(df['subject_id']) == ['12', '57']  # must have right values, in order matching id_strs


def test_show_coverage(sim_model, save_dir):
    sim_model.show_coverage(save_dir)
    assert os.path.exists(os.path.join(save_dir, 'discrete_coverage.png'))
