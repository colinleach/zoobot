import logging
import os

import pandas as pd


def filename_to_dict(filename):
    assert filename[-5:] == '.fits'
    filename_components = filename.rstrip('.fits').split('_')

    data = {}
    data['id'] = str(filename_components[2])
    data['band'] = int(filename_components[4])
    data['camera'] = int(filename_components[6])
    data['background'] = int(filename_components[8])

    return data


def read_catalog_from_dir(dir, name):
    filenames = list(filter(lambda x: x[-5:] == '.fits', os.listdir(dir)))
    catalog = pd.DataFrame(list(map(lambda x: filename_to_dict(x), filenames)))
    catalog['fits_loc'] = list(map(lambda x: os.path.join(dir, x), filenames))
    catalog['name'] = name

    logging.info('Images identified in folder {} and assigned type {}: {}'.format(
        dir, name, len(catalog)))
    logging.info('Galaxies identified in folder {} and assigned type {}: {}'.format(
        dir, name, len(catalog['id'].unique())))
    return catalog


if __name__ == '__main__':

    logging.basicConfig(
        filename='make_catalog.log',
        format='%(levelname)s:%(message)s',
        filemode='w',
        level=logging.INFO)

    catalog_root_dir = '/data/onedrive/Shared/Illustris'

    available_catalogs = [
        {'name': 'major_merger',
         'dir': os.path.join(catalog_root_dir, 'MajorMergers'),
         },

        {'name': 'minor_merger',
         'dir': os.path.join(catalog_root_dir, 'MinorMergers'),
         },

        {'name': 'non_merger',
         'dir': os.path.join(catalog_root_dir, 'NonMergers'),
         },

        {'name': 'random_sample',
         'dir': os.path.join(catalog_root_dir, 'RandomSample1'),
         },

        {'name': 'random_sample',
         'dir': os.path.join(catalog_root_dir, 'RandomSample2'),
         },
    ]

    catalogs = [read_catalog_from_dir(catalog['dir'], name=catalog['name']) for catalog in available_catalogs]
    # TODO check no overlaps between catalogs
    # if any galaxies are in both random sample and another folder, remove from random sample
    full_catalog = pd.concat(catalogs)

    logging.info('Images identified from directories: {}'.format(
        len(full_catalog)))
    logging.info('Galaxies identified from directories: {}'.format(
        len(full_catalog['id'].unique())))

    # add in metadata from Tim on mass, merger time, etc.
    catalog_metadata = pd.read_csv(os.path.join(catalog_root_dir, 'galaxies_data.txt'), dtype={'id': str}).drop_duplicates()
    assert all(catalog_metadata['id'].unique())
    logging.info('Metadata entries: {}'.format(
        len(catalog_metadata)))

    assert set(full_catalog['id']).issubset(set(catalog_metadata['id']))
    # full_catalog_with_metadata = pd.merge(full_catalog, catalog_metadata, on='id', how='outer')
    # TODO temporarily, we drop duplicate metadata entries
    full_catalog_with_metadata = pd.merge(
        full_catalog,
        catalog_metadata.drop_duplicates(subset=['id'], keep='first'),
        on='id',
        how='inner')

    catalog_loc = 'catalogs/illustris_galaxies.csv'
    full_catalog_with_metadata.reset_index().to_csv(catalog_loc)
    logging.info('Saved {} galaxies, {} images to {}'.format(
        len(full_catalog_with_metadata['id'].unique()),
        len(full_catalog_with_metadata),
        catalog_loc)
    )