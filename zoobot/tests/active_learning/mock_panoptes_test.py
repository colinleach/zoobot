import pytest

import pandas as pd

from zoobot.active_learning import mock_panoptes


def test_get_labels(monkeypatch):

    def mock_read_csv(oracle_loc, usecols, dtype):
        return pd.DataFrame([
            {
                'id_str': '20927311',
                'label': 0.6
            },
            {
                'id_str': '20530807',
                'label': 0.3
            }
        ])

    monkeypatch.setattr(mock_panoptes.pd, 'read_csv', mock_read_csv)

    subject_ids = ['20927311', '20530807']
    labels = mock_panoptes.get_labels(subject_ids)
    assert labels == [0.6, 0.3]
