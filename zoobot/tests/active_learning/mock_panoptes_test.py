import pytest

import os
import json

import pandas as pd

from zoobot.active_learning import mock_panoptes


@pytest.fixture()
def oracle_loc(tmpdir):
    df = pd.DataFrame([
        {
            'id_str': '20927311',
            'label': 12,
            'total_votes': 3

        },
        {
            'id_str': '20530807',
            'label': 4,
            'total_votes': 2
        }
    ])
    oracle_dir = tmpdir.mkdir('oracle_dir').strpath
    oracle_loc = os.path.join(oracle_dir, 'oracle.csv')
    df.to_csv(oracle_loc)
    return oracle_loc


@pytest.fixture(params=[True, False])
def subjects_requested_save_loc_possible(request, subjects_to_request, subjects_requested_save_loc):
    if request.param:
        json.dump(subjects_to_request, open(subjects_requested_save_loc, 'w'))
        return subjects_requested_save_loc
    else:
        return 'broken_path'


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
    


def test_get_labels(subjects_requested_save_loc_possible, subjects_to_request, oracle_loc):

    subject_ids, labels, counts = mock_panoptes.get_labels(oracle_loc)

    if subjects_requested_save_loc_possible == 'broken_path':
        assert subject_ids == []
        assert labels == []
    else:
        assert isinstance(subject_ids, list)
        assert isinstance(labels, list)
        assert isinstance(counts, list)
        assert subject_ids == subjects_to_request
        assert labels == [12, 4]
        assert counts == [3, 2]
        # should have been subsequently deleted
        assert not os.path.exists(subjects_requested_save_loc_possible)
