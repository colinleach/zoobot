

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy import units as u

from shared_utilities import plot_catalog_overlap


def get_spiral_classification_results(published_data_loc, nrows=None):
    """
    Get spiral classification data table from GZ2 published results
    https://data.galaxyzoo.org/ Galaxy Zoo 2 Table 1
    Direct link: http://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz

    Args:
        nrows (int): num. of rows to return (from head). If None, return all rows.

    Returns:
        (pd.DataFrame) DR7 ID, spiral weighted vote fraction, and vote counts for spiral/not spiral, by subject
    """

    is_spiral_col = 't04_spiral_a08_spiral_weighted_fraction'
    spiral_count_col = 't04_spiral_a08_spiral_count'
    no_spiral_count_col = 't04_spiral_a09_no_spiral_count'

    useful_columns = ['dr7objid', 'ra', 'dec', is_spiral_col, spiral_count_col, no_spiral_count_col]

    df = pd.read_csv(published_data_loc, nrows=nrows, usecols=useful_columns)

    return df


def match_galaxies_to_catalog(galaxies, catalog, matching_radius=10 * u.arcsec):
    # http://docs.astropy.org/en/stable/coordinates/matchsep.html

    galaxies_coord = SkyCoord(ra=galaxies['ra'] * u.degree, dec=galaxies['dec'] * u.degree)
    catalog_coord = SkyCoord(ra=catalog['ra'] * u.degree, dec=catalog['dec'] * u.degree)
    best_match_catalog_index, sky_separation, _ = galaxies_coord.match_to_catalog_sky(catalog_coord)

    galaxies['best_match'] = best_match_catalog_index
    galaxies['sky_separation'] = sky_separation.to(u.arcsec).value
    matched_galaxies = galaxies[galaxies['sky_separation'] < matching_radius.value]

    catalog['best_match'] = catalog.index.values

    matched_catalog = pd.merge(matched_galaxies, catalog, on='best_match', how='inner', suffixes=['_subject', ''])

    return matched_catalog


if __name__ == '__main__':

    nrows = None

    catalog_dir = '/data/galaxy_zoo/gz2/subjects'
    published_data_loc = '{}/gz2_hart16.csv'.format(catalog_dir)  # volunteer labels
    subject_manifest_loc = '{}/galaxyzoo2_sandor.csv'.format(catalog_dir)  # subjects on AWS

    labels_loc = '{}/all_labels.csv'.format(catalog_dir)  # will place catalog of file list and labels here

    # get the useful spiral label columns for each subject
    spiral_results = get_spiral_classification_results(published_data_loc, nrows=nrows)
    print('Published subjects with labels: {}'.format(len(spiral_results)))

    subject_manifest = pd.read_csv(subject_manifest_loc, nrows=nrows)
    print('AWS subjects from Sandor: {}'.format(len(subject_manifest)))

    plot_catalog_overlap(
        spiral_results,
        subject_manifest,
        ['published subjects', 'sandor AWS subjects'],
        'sandor_and_volunteers_overlap')

    catalog = match_galaxies_to_catalog(spiral_results, subject_manifest)

    assert len(catalog) > 0

    catalog.to_csv(labels_loc)
