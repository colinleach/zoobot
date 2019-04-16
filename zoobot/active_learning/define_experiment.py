import os
import shutil
import logging
import argparse

import pandas as pd

"""Shared logic to refine catalogs"""

def get_experiment_catalogs(catalog, question, save_dir):
    catalog = shuffle(catalog)  # crucial for GZ2!
    catalog = define_identifiers(catalog)
    labelled, unlabelled = split_labelled_and_unlabelled(catalog, question)
    labelled = define_labels(labelled, question)  # unlabelled and catalog have no 'label' column
    return catalog, labelled, unlabelled


def split_labelled_and_unlabelled(catalog, question):
    retired = catalog.apply(subject_is_retired, axis=1)
    labelled = catalog[retired]
    unlabelled= catalog[~retired]
    return labelled, unlabelled


def shuffle(df):
    # THIS IS CRUCIAL. GZ catalog is not properly shuffled, and featured-ness changes systematically
    return df.sample(len(df)).reset_index()


def define_identifiers(catalog):
    if any(catalog.duplicated(subset=['iauname'])):
        logging.warning('Found duplicated iaunames - dropping!')
        catalog = catalog.drop_duplicates(subset=['iauname'], keep=False)
    catalog['id_str'] = catalog['iauname']
    return catalog


def subject_is_retired(subject):
    return subject['smooth-or-featured_total-votes'] > 36


def define_labels(labelled, question):
    labelled['total_votes'] = labelled['smooth-or-featured_total-votes'].astype(int)
    assert all(labelled['total_votes'] > 0)  # otherwise, can't be labelled! and will cause nans
    if question == 'smooth':
        labelled['label'] = labelled['smooth-or-featured_smooth'].astype(int)
    elif question == 'bar':
        labelled['total_votes'] = labelled['bar_total-votes'].astype(int)
        labelled = labelled[labelled['total_votes'] > 10]  # drop anything with n < 10
        try:
            labelled['label'] = labelled['bar_weak'].astype(int)  # DECALS
        except KeyError:
            labelled['label'] = labelled['bar_yes'].astype(int)  # GZ2
    else:
        raise ValueError('question {} not understood'.format(question))

    return labelled



def get_mock_catalogs(labelled_catalog, save_dir, labelled_size=15000):
    # given a (historical) labelled catalog, pretend split into labelled and unlabelled
    assert not any(pd.isnull(labelled_catalog['label']))
    oracle = labelled_catalog[['id_str', 'total_votes', 'label']]
    mock_labelled = labelled_catalog[:labelled_size]  # for training and eval. Could do basic split on these!
    mock_unlabelled = labelled_catalog[labelled_size:]  # for pool
    del mock_unlabelled['label']
    return mock_labelled, mock_unlabelled, oracle



if __name__ == '__main__':

    # master_catalog_loc = 'data/decals/decals_master_catalog.csv'  # currently with all galaxies but only a few classifications
    # catalog_dir = 'data/decals/prepared_catalogs/{}'.format(name)

    parser = argparse.ArgumentParser(description='Define experiment (labelled catalog, question, etc) from master catalog')
    parser.add_argument('--master-catalog', dest='master_catalog_loc', type=str,
                    help='Name of experiment (save to data')
    parser.add_argument('--question', dest='question', type=str,
                    help='Question to answer: smooth or bar')
    parser.add_argument('--save-dir', dest='save_dir', type=str,
                    help='Save experiment catalogs here')
    args = parser.parse_args()
    master_catalog_loc = args.master_catalog_loc
    question = args.question
    save_dir = args.save_dir

    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    master_catalog = pd.read_csv(master_catalog_loc)
    catalog, labelled, unlabelled = get_experiment_catalogs(master_catalog, question, save_dir)

    # ad hoc filtering here
    # catalog = catalog[:20000]

    labelled.to_csv(os.path.join(save_dir, 'labelled_catalog.csv'), index=False)
    unlabelled.to_csv(os.path.join(save_dir, 'unlabelled_catalog.csv'), index=False)
    catalog.to_csv(os.path.join(save_dir, 'full_catalog.csv'), index=False)

    simulation_dir = os.path.join(save_dir, 'simulation_context')
    if not os.path.isdir(simulation_dir):
        os.mkdir(simulation_dir)
    mock_labelled, mock_unlabelled, oracle = get_mock_catalogs(labelled, simulation_dir)

    mock_labelled.to_csv(os.path.join(simulation_dir, 'labelled_catalog.csv'), index=False)
    oracle.to_csv(os.path.join(simulation_dir, 'oracle.csv'), index=False)
    mock_unlabelled.to_csv(os.path.join(simulation_dir, 'unlabelled_catalog.csv'), index=False)
