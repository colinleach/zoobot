import pytest

import os

from zoobot.illustris import make_catalog


@pytest.fixture()
def filename():
    return 'synthetic_image_104798_band_5_camera_0_bg_1.fits'


@pytest.fixture()
def dir(tmpdir):

    dir_object = tmpdir.mkdir('illustris_test_dir')
    dir_path = dir_object.strpath

    # fill directory with fake examples
    filenames = [
        'synthetic_image_104798_band_5_camera_0_bg_1.fits',
        'synthetic_image_104798_band_5_camera_1_bg_1.fits',  # different camera
        'synthetic_image_104798_band_5_camera_2_bg_1.fits',  # different camera

        'synthetic_image_204798_band_5_camera_0_bg_1.fits',  # different id

        'something_else.else'  # different file type, not to be included
    ]
    file_locs = [os.path.join(dir_path, file) for file in filenames]

    for file_loc in file_locs:
        with open(file_loc, mode='w') as file:
            file.write('I am a fake fits image')

    assert os.listdir(dir_path) == filenames

    return dir_path


def test_filename_to_dict(filename):
    data = make_catalog.filename_to_dict(filename)
    assert data['id'] == '104798'  # treat id as a string
    assert data['band'] == 5
    assert data['camera'] == 0
    assert data['background'] == 1


def test_read_catalog_from_dir(dir):
    catalog = make_catalog.read_catalog_from_dir(dir, 'test_catalog')
    print(catalog)
    assert len(catalog) == 4
    assert all(catalog['name'] == 'test_catalog')