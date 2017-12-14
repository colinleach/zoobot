import pytest

from get_galaxy_zoo import *


@pytest.fixture()
def catalog():
    return pd.DataFrame([
        {'ra': 12.,
         'dec': 13.},

        {'ra': 12.,
         'dec': 16.}
    ])


@pytest.fixture()
def galaxies():
    return pd.DataFrame([
        # matches a catalog entry
        {'ra': 12.00001,
         'dec': 13.},

        # doesn't match
        {'ra': 80.,
         'dec': 80.}
    ])


def test_get_galaxy_zoo(galaxies, catalog):
    matching_radius = 10 * u.arcsec
    matched = match_galaxies_to_catalog(galaxies, catalog, matching_radius=matching_radius)
    print(matched)
    assert len(matched) == 1
