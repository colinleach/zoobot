import pytest

import os
import json

import pandas as pd

from zoobot.active_learning import mock_panoptes


@pytest.fixture()
def subjects_requested_save_loc(monkeypatch, tmpdir):
    save_loc = os.path.join(tmpdir.mkdir('temp').strpath, 'subjects_requested.json')
    monkeypatch.setattr(mock_panoptes, 'SUBJECTS_REQUESTED', save_loc)
    return save_loc


@pytest.fixture()
def subjects_to_request():
    return ['20927311', '20530807']


def test_request_labels(subjects_requested_save_loc, subjects_to_request):
    mock_panoptes.request_labels(subjects_to_request)
    requested_subjects = json.load(open(subjects_requested_save_loc, 'r'))
    assert requested_subjects == subjects_to_request
    


def test_get_labels(monkeypatch, subjects_requested_save_loc, subjects_to_request):

    json.dump(subjects_to_request, open(subjects_requested_save_loc, 'w'))

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

    subject_ids, labels = mock_panoptes.get_labels()

    assert not os.path.exists(subjects_requested_save_loc)
    assert subject_ids == subjects_to_request
    assert labels == [0.6, 0.3]
