import pytest

import pandas as pd

from zoobot.illustris import illustris_to_tfrecord


@pytest.fixture()
def catalog():
    return pd.DataFrame([
        {'id': '0',
         'name': 'major_merger'
         },

        {'id': '1',
         'name': 'major_merger'
         },

        {'id': '2',
         'name': 'minor_merger'
         },

        {'id': '3',
         'name': 'minor_merger'
         },

        {'id': '4',
         'name': 'non_merger'
         },

        {'id': '5',
         'name': 'non_merger'
         },

        {'id': '0',  # should be removed
         'name': 'random_sample'
         },

        {'id': '2',  # should be removed
         'name': 'random_sample'
         },

        {'id': '4',  # should be removed
         'name': 'random_sample'
         },

        {'id': '9',
         'name': 'random_sample'
         },

    ])


@pytest.fixture()
def fits_loc():
    return 'zoobot/test_examples/illustris_test_dir/synthetic_image_104798_band_5_camera_0_bg_1.fits'


@pytest.fixture()
def subject(fits_loc):
    return {'fits_loc': fits_loc}


def test_render_fits(subject):
    pil_img = illustris_to_tfrecord.load_illustris_as_pil(subject)
    # pil_img.show()


def test_remove_mergers_from_random_sample(catalog):
    clean_catalog = illustris_to_tfrecord.remove_known_galaxies_from_random_sample(catalog)
    print(clean_catalog['id'])
    assert all(clean_catalog['id'].values == ['0', '1', '2', '3', '4', '5', '9'])
