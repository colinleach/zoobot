import os
import shutil
import logging
import argparse
import glob

import pandas as pd

from zoobot import label_metadata

"""Shared logic to refine catalogs. All science decisions should have happened after this script"""

def get_experiment_catalogs(catalog, save_dir, filter_catalog=False):
    catalog = shuffle(catalog)  # crucial for GZ2!
    catalog = define_identifiers(catalog)
    if filter_catalog:
        filtered_catalog = apply_custom_filter_cheat(catalog)  # science logic lives here
    else:
        filtered_catalog = catalog
    labelled, unlabelled = split_retired_and_not(filtered_catalog)  # for now using N=36, ignore galaxies with less labels
    # unlabelled and catalog will have no 'label' column
    return catalog, labelled, unlabelled


def split_retired_and_not(catalog):
    retired = catalog.apply(subject_is_retired, axis=1)
    labelled = catalog[retired]
    unlabelled = catalog[~retired]
    return labelled, unlabelled


def shuffle(df):
    # THIS IS CRUCIAL. GZ catalog is not properly shuffled, and featured-ness changes systematically
    return df.sample(len(df)).reset_index()


def define_identifiers(catalog):
    if any(catalog.duplicated(subset=['id_str'])):
        logging.warning('Found duplicated iaunames - dropping!')
        catalog = catalog.drop_duplicates(subset=['id_str'], keep=False)
    return catalog


def apply_custom_filter_ml(catalog):
    # expect to change this a lot
    # for now, expect a 'smooth-or-featured_featured-or-disk_prediction_mean' col and filter > 0.25 (i.e. 10 of 40)
    # predicted beforehand by existing model TODO

    min_featured = 0.5  # see errors.ipynb
    # previous_prediction_locs = glob.glob('temp/master_256_predictions_*.csv')
    # previous_predictions = pd.concat([pd.read_csv(loc, usecols=['id_str', 'smooth-or-featured_featured-or-disk_prediction_mean']) for loc in previous_prediction_locs])
    previous_predictions = pd.read_csv('temp/smooth_or_featured_labelled_latest_with_edge.csv', usecols=['id_str', 'smooth-or-featured_featured-or-disk_prediction_mean', 'disk-edge-on_yes_prediction_mean_dummy'])
    len_before = len(catalog)
    catalog = pd.merge(catalog, previous_predictions, on='id_str', how='inner')
    len_merged = len(catalog)
    # print(catalog['smooth-or-featured_featured-or-disk_prediction_mean'])
    featured = catalog[catalog['smooth-or-featured_featured-or-disk_prediction_mean'] > min_featured]
    filtered_catalog = featured[featured['disk-edge-on_yes_prediction_mean_dummy'] < 0.5]
    logging.info(f'{len_before} before filter, {len_merged} after merge, {len(filtered_catalog)} after filter at min_featured={min_featured}')
    print(f'{len_before} before filter, {len_merged} after merge, {len(filtered_catalog)} after filter at min_featured={min_featured}')
    return filtered_catalog

# duplicate in make_decals_tfrecords.py
def apply_custom_filter_cheat(catalog):
    min_featured = 0.5  # will be a bit different, volunteers here
    is_featured = (catalog['smooth-or-featured_featured-or-disk'] / catalog['smooth-or-featured_total-votes']) > min_featured
    featured = catalog[is_featured]
    is_face_on = (featured['disk-edge-on_yes'] / featured['smooth-or-featured_featured-or-disk']) < 0.5
    catalog = featured[is_face_on]
    return catalog


def subject_is_retired(subject):
    return subject['smooth-or-featured_total-votes'] > 36


def drop_duplicates(df):
    #  to be safe, could improve TODO
    if any(df['id_str'].duplicated()):
        logging.warning('Duplicated:')
        counts = df['id_str'].value_counts()
        logging.warning(counts[counts > 1])
    # no effect if no duplicates
    return df.drop_duplicates(subset=['id_str'], keep=False)


def get_mock_catalogs(labelled_catalog, save_dir, labelled_size, label_cols):
    # given a (historical) labelled catalog, pretend split into labelled and unlabelled
    for label_col in label_cols:
        if any(pd.isnull(labelled_catalog[label_col])):
            logging.critical(labelled_catalog[label_col])
            raise ValueError(f'Missing at least one label for {label_col}')
    # oracle has everything in real labelled catalog
    oracle = labelled_catalog.copy()  # could filter cols here if needed
    mock_labelled = labelled_catalog[:labelled_size]  # for training and eval
    mock_unlabelled = labelled_catalog[labelled_size:]  # for pool
    for label_col in label_cols:
        del mock_unlabelled[label_col]
    return mock_labelled, mock_unlabelled, oracle


if __name__ == '__main__':

    """
    
    Decals: see dvc.md

    GZ2: python zoobot/science_logic/define_experiment.py --master-catalog data/gz2/gz2_master_catalog.csv --save-dir data/gz2/prepared_catalogs/all_featp5_facep5_2p5 --sim-fraction 2.5 --filter
    python zoobot/science_logic/define_experiment.py --master-catalog data/gz2/gz2_master_catalog.csv --save-dir data/gz2/prepared_catalogs/all_2p5_unfiltered --sim-fraction 2.5

    $PYTHON zoobot/science_logic/define_experiment.py --master-catalog data/gz2/gz2_master_catalog_arc.csv --save-dir data/gz2/prepared_catalogs/all_arc_unfiltered --sim-fraction 2.5
    """

    # master_catalog_loc = 'data/decals/decals_master_catalog.csv'  # currently with all galaxies but only a few classifications
    # catalog_dir = 'data/decals/prepared_catalogs/{}'.format(name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description='Define experiment (labelled catalog, question, etc) from master catalog')
    parser.add_argument('--master-catalog', dest='master_catalog_loc', type=str,
                        help='Name of experiment (save to data')
    parser.add_argument('--save-dir', dest='save_dir', type=str,
                        help='Save experiment catalogs here')
    parser.add_argument('--sim-fraction', dest='sim_fraction', type=float, default=4.,
                        help='Save experiment catalogs here')
    parser.add_argument('--filter', dest='filter_catalog', action='store_true', default=False)
    args = parser.parse_args()
    master_catalog_loc = args.master_catalog_loc
    save_dir = args.save_dir

    # label_cols = label_metadata.decals_partial_label_cols
    # label_cols = label_metadata.gz2_partial_label_cols
    label_cols = label_metadata.gz2_label_cols

    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    master_catalog = pd.read_csv(master_catalog_loc)
    catalog, labelled, unlabelled = get_experiment_catalogs(
        master_catalog, save_dir, filter_catalog=args.filter_catalog)

    # ad hoc filtering here
    # catalog = catalog[:20000]

    labelled.to_csv(os.path.join(
        save_dir, 'labelled_catalog.csv'), index=False)
    unlabelled.to_csv(os.path.join(
        save_dir, 'unlabelled_catalog.csv'), index=False)
    catalog.to_csv(os.path.join(save_dir, 'full_catalog.csv'), index=False)

    simulation_dir = os.path.join(save_dir, 'simulation_context')
    if not os.path.isdir(simulation_dir):
        os.mkdir(simulation_dir)

    labelled_size = int(len(labelled) / args.sim_fraction)  # pretend unlabelled, to be acquired
    mock_labelled, mock_unlabelled, oracle = get_mock_catalogs(
        labelled, simulation_dir, labelled_size, label_cols)

    mock_labelled.to_csv(os.path.join(
        simulation_dir, 'labelled_catalog.csv'), index=False)
    oracle.to_csv(os.path.join(simulation_dir, 'oracle.csv'), index=False)
    mock_unlabelled.to_csv(os.path.join(
        simulation_dir, 'unlabelled_catalog.csv'), index=False)
