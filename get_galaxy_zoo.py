import os

import pandas as pd
from pymongo import MongoClient
from astropy.coordinates import SkyCoord
from astropy import units as u

import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image

pd.options.display.max_rows = 200
pd.options.display.max_columns = 100


def plot_catalog_overlap(catalog_a, catalog_b, legend):

    a_coords = catalog_a[['ra', 'dec']]
    a_coords['catalog'] = legend[0]
    b_coords = catalog_b[['ra', 'dec']]
    b_coords['catalog'] = legend[1]

    df_to_plot = pd.concat([a_coords, b_coords])
    df_to_plot['catalog'] = df_to_plot['catalog'].astype('category')

    canvas = ds.Canvas(plot_width=300, plot_height=300)
    aggc = canvas.points(df_to_plot, 'ra', 'dec', ds.count_cat('catalog'))
    img = tf.shade(aggc)
    export_image(img, 'catalog_overlap')


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


def get_subject_manifest(subject_manifest_loc, nrows=None):
    """

    Args:
        nrows ():

    Returns:

    """

    df = pd.read_csv(subject_manifest_loc, nrows=nrows)

    return df


def create_subject_manifest(subject_manifest_loc, survey_tag):
    """
    Convert Galaxy Zoo 2 classifications export into subject manifest
    subject manifest is list of subjects with id, ra, dec

    Args:
        classifications_loc (str): absolute path to classifications export
        subject_manifest_loc (str): absolute path to save subject manifest

    Returns:
        None
    """

    client = MongoClient()
    db = client.galaxy_zoo

    subjects = db.subjects

    if survey_tag == 'all':
        cursor = subjects.find()
    else:
        cursor = subjects.find({'metadata.survey': survey_tag})

    sloan_data = []
    for document in cursor:
        galaxy = {
            'zooniverse_id': document['zooniverse_id'],
            'ra': document['coords'][0],
            'dec': document['coords'][1],
            'img_server_loc': document['location']['standard']
        }
        sloan_data.append(galaxy)

    df = pd.DataFrame(sloan_data)
    df.index.name = 'index'
    df.to_csv('/data/galaxy_zoo/gz2/subjects/{}_subjects.csv'.format(survey_tag))


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
    # gz_catalog = pd.read_csv('/data/galaxy_zoo/decals/catalogs/nsa_all_raw_gz_counts_10.0_arcsec.csv', nrows=1000)
    # print(gz_catalog.iloc[0])

    nrows = None
    survey_tag = 'all'

    catalog_dir = '/data/galaxy_zoo/gz2/subjects'
    published_data_loc = '{}/gz2_hart16.csv'.format(catalog_dir)
    classifications_loc = '{}/galaxy_zoo_original_classifications.csv'.format(catalog_dir)
    # subject_manifest_loc = '{}/{}_subjects.csv'.format(catalog_dir, survey_tag)
    subject_manifest_loc = '{}/galaxyzoo2_sandor.csv'.format(catalog_dir)

    labels_loc = '{}/{}_labels.csv'.format(catalog_dir, survey_tag)

    if not os.path.exists(subject_manifest_loc):
        create_subject_manifest(subject_manifest_loc, survey_tag)  # requires mongodb

    spiral_results = get_spiral_classification_results(published_data_loc, nrows=nrows)
    print('published subjects: {}'.format(len(spiral_results)))

    subject_manifest = get_subject_manifest(subject_manifest_loc, nrows=nrows)
    print('server subjects: {}'.format(len(subject_manifest)))

    plot_catalog_overlap(spiral_results, subject_manifest, ['published', 'subjects'])

    catalog = match_galaxies_to_catalog(spiral_results, subject_manifest)

    print(len(catalog))

    # canvas = ds.Canvas(plot_width=300, plot_height=300)
    # aggc = canvas.points(spiral_results, 'ra', 'dec')
    # img = tf.shade(aggc)
    # export_image(img, 'spiral')

    canvas = ds.Canvas(plot_width=300, plot_height=300)
    aggc = canvas.points(subject_manifest, 'ra', 'dec')
    img = tf.shade(aggc)
    export_image(img, 'subjects_sandor')

    canvas = ds.Canvas(plot_width=300, plot_height=300)
    aggc = canvas.points(catalog, 'ra', 'dec')
    img = tf.shade(aggc)
    export_image(img, 'matched')

    catalog.to_csv(labels_loc)
