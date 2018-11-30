import os

import pandas as pd
import pytest

from zoobot.tests import TEST_EXAMPLE_DIR


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
    return os.path.join(TEST_EXAMPLE_DIR, 'synthetic_image_104798_band_5_camera_0_bg_1.fits')


@pytest.fixture()
def subject(fits_loc):
    return {'fits_loc': fits_loc}
