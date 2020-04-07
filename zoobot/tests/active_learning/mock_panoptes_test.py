import pytest

import os
import json

import pandas as pd

from zoobot.active_learning import oracles


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
        },
        {
            'id_str': '20530808',
            'label': 4,
            'total_votes': 2
        },
        {
            'id_str': '205308089',
            'label': 7,
            'total_votes': 2
        }
    ])
    oracle_dir = tmpdir.mkdir('oracle_dir').strpath
    oracle_loc = os.path.join(oracle_dir, 'oracle.csv')
    df.to_csv(oracle_loc)
    return oracle_loc


@pytest.fixture()
def subjects_to_request():
    return ['20927311', '20530807']


@pytest.fixture()
def previously_requested_subjects():
    return ['20530808', '205308089']

@pytest.fixture(params=[True, False])
def previously_requested_subjects_exist(request):
    return request.param

# Mock Panoptes tests

@pytest.fixture()
def subjects_requested_loc(tmpdir, previously_requested_subjects, previously_requested_subjects_exist):
    loc = os.path.join(tmpdir.mkdir('temp').strpath, 'subjects_requested.json')
    if previously_requested_subjects_exist:
        with open(loc, 'w') as f:
            json.dump(previously_requested_subjects, f)
    # otherwise, path will be okay but no file will be there
    return loc    

@pytest.fixture()
def panoptes_mock(oracle_loc, subjects_requested_loc):
    return oracles.PanoptesMock(
        oracle_loc=oracle_loc,
        subjects_requested_loc=subjects_requested_loc)

def test_panoptes_mock_with_bad_oracle_loc(oracle_loc, subjects_requested_loc):
    with pytest.raises(AssertionError):
        oracles.PanoptesMock('broken_oracle_loc', subjects_requested_loc)

def test_request_labels(panoptes_mock, subjects_to_request):
    panoptes_mock.request_labels(subjects_to_request, 'dummy_uploader_name', retirement=40)
    requested_subjects = json.load(open(panoptes_mock._subjects_requested_loc, 'r'))
    assert requested_subjects == subjects_to_request
    

def test_get_labels(panoptes_mock, previously_requested_subjects, previously_requested_subjects_exist):

    if not previously_requested_subjects_exist:
        # should have never been made (check the test setup)
        assert not os.path.isfile(panoptes_mock._subjects_requested_loc)

    subject_ids, labels, counts = panoptes_mock.get_labels()

    if previously_requested_subjects_exist:
        assert isinstance(subject_ids, list)
        assert isinstance(labels, list)
        assert isinstance(counts, list)
        assert subject_ids == previously_requested_subjects
        assert labels == [4, 7]
        assert counts == [2, 2]
        # should have been subsequently deleted
        assert not os.path.isfile(panoptes_mock._subjects_requested_loc)
    else:
        
        assert subject_ids == []
        assert labels == []


# (Real) Panoptes tests


@pytest.fixture()
def catalog_loc(tmpdir):
    df = pd.DataFrame([
        {
            'subject_id': '20927311',
            'ra': 12.0,
            'dec': 14.0

        },
        {
            'subject_id': '20530807',
            'ra': 12.0,
            'dec': 14.0
        },
        {
            'subject_id': '20530808',
            'ra': 12.0,
            'dec': 14.0
        },
        {
            'subject_id': '205308089',
            'ra': 12.0,
            'dec': 14.0
        }
    ])
    catalog_dir = tmpdir.mkdir('catalog_dir').strpath
    catalog_loc = os.path.join(catalog_dir, 'oracle.csv')
    df.to_csv(catalog_loc)
    return catalog_loc


@pytest.fixture()
def panoptes(catalog_loc):
    return oracles.Panoptes(
        catalog_loc=catalog_loc,
        login_loc='some_login.txt',
        project_id='1234')

def test_panoptes_request_labels(mocker, panoptes, subjects_to_request):
    mocker.patch('shared_astro_utils.upload_utils.create_manifest_from_catalog', autospec=True)
    mocker.patch('shared_astro_utils.upload_utils.upload_manifest_to_galaxy_zoo', autospec=True)

    panoptes.request_labels(subjects_to_request, 'test_request')

    from shared_astro_utils.upload_utils import create_manifest_from_catalog, upload_manifest_to_galaxy_zoo

    assert create_manifest_from_catalog.call_args[0][0].equals(panoptes._full_catalog[panoptes._full_catalog['subject_id'].isin(subjects_to_request)])
    upload_args = upload_manifest_to_galaxy_zoo.call_args[1]
    assert isinstance(upload_args['subject_set_name'], str)
    assert upload_args['galaxy_zoo_id'] == '1234'
    assert upload_args['login_loc'] == 'some_login.txt'

