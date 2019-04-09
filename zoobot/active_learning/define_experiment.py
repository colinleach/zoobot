import os
import shutil
import logging

import pandas as pd

"""Shared logic to refine catalogs"""

def experiment_catalog(catalog, question, save_dir):
    catalog = shuffle(catalog)  # crucial for GZ2!
    catalog = define_identifiers(catalog)
    catalog = define_labels(catalog, question)
    return catalog


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
    return subject['smooth-or-featured_total-votes'] > 1 # TODO 37  


def define_labels(catalog, question):
    catalog['total_votes'] = catalog['smooth-or-featured_total-votes']
    if question == 'smooth':
        catalog['label'] = catalog['smooth-or-featured_smooth']
    elif question == 'bar':
        catalog['total_votes'] = catalog['bar_total-votes']
        try:
            catalog['label'] = catalog['bar_weak']  # DECALS
        except KeyError:
            catalog['label'] = catalog['bar_yes']  # GZ2
    else:
        raise ValueError('question {} not understood'.format(question))
    return catalog


def save_catalog(catalog, save_dir):
    retired = catalog.apply(subject_is_retired, axis=1)
    labelled_catalog = catalog[retired]
    unlabelled_catalog = catalog[~retired]

    labelled_catalog.to_csv(os.path.join(save_dir, 'labelled_catalog.csv'), index=False)
    unlabelled_catalog.to_csv(os.path.join(save_dir, 'unlabelled_catalog.csv'), index=False)
    catalog.to_csv(os.path.join(save_dir, 'full_catalog.csv'), index=False)


def save_mock_catalog(catalog, save_dir, train_size=256, eval_size=2500):
    # given a (historical) catalog, pretend split into labelled and unlabelled
    labelled_size = train_size + eval_size
    labelled_catalog = catalog[:labelled_size]  # for training and eval. Could do basic split on these!
    unlabelled_catalog = catalog[labelled_size:]  # for pool
    del unlabelled_catalog['label']

    labelled_catalog.to_csv(os.path.join(save_dir, 'labelled_catalog.csv'), index=False)
    labelled_catalog[['id_str', 'total_votes', 'label']].to_csv(os.path.join(save_dir, 'oracle.csv'), index=False)
    unlabelled_catalog.to_csv(os.path.join(save_dir, 'unlabelled_catalog.csv'), index=False)


if __name__ == '__main__':

    name = 'smooth_unfiltered'
    question = 'smooth'

    master_catalog_loc = 'data/decals/decals_master_catalog.csv'
    catalog_dir = 'data/decals/prepared_catalogs/{}'.format(name)
    if os.path.isdir(catalog_dir):
        shutil.rmtree(catalog_dir)
    os.mkdir(catalog_dir)

    master_catalog = pd.read_csv(master_catalog_loc)
    catalog = experiment_catalog(master_catalog, question, catalog_dir)

    # ad hoc filtering here
    catalog = catalog[:10000]
    if question == 'bar':  # filter to at least a bit featured
        catalog = catalog[catalog['bar_total-votes'] > 10]  

    save_catalog(catalog, catalog_dir)

    simulation_dir = os.path.join(catalog_dir, 'simulation_context')
    if not os.path.isdir(simulation_dir):
        os.mkdir(simulation_dir)
    save_mock_catalog(catalog, simulation_dir)
