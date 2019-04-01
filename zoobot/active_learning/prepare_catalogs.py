# load the decals joint catalog and save only the important columns
import logging
import os

import pandas as pd
from astropy.table import Table

def tweak_joint_catalog(catalog_loc, save_loc):
    """rename and resave a few joint catalog columns, to make uploading easier"""
    joint_catalog_table = Table.read(catalog_loc)
    # print(joint_catalog_table.colnames)
    df = joint_catalog_table.to_pandas()
    # conversion messes up strings into bytes
    for str_col in ['iauname', 'png_loc', 'fits_loc']:
        df[str_col] = df[str_col].apply(lambda x: x.decode('utf-8'))
    df['nsa_version'] = 'v1_0_0'
    df = df.rename(index=str, columns={
        'fits_loc': 'local_fits_loc',
        'png_loc': 'local_png_loc',
        'z': 'redshift'
    })
    print(df.iloc[0])
    df['png_loc'] = df['local_png_loc'].apply(lambda x: 'data/' + x.lstrip('/Volumes/alpha'))  # change to be inside data folder, specified relative to repo root
    print(df.iloc[0]['png_loc'])
    print('Galaxies: {}'.format(len(df)))
    df.to_csv(save_loc, index=False)


"""Logic to turn GZ2 or DECALS previous outputs into shared-schema catalogs (labelled or otherwise) for AL"""


def tweak_previous_decals_classifications(catalog_loc, save_loc):
    # create a labelled catalog to use when starting all iterations
    # this will be used to make the first training shard
    catalog = pd.read_csv(catalog_loc)
    catalog['id_str'] = catalog['subject_id'].astype(str)
    catalog.to_csv(save_loc, index=False)

# TODO decals predictions need to be joined with the joint catalog to work out where the files are. I should think carefully about the best place for this to happen.

def tweak_previous_gz2_classifications(catalog_loc, save_loc):
    usecols = [
        't01_smooth_or_features_a01_smooth_count',
        't01_smooth_or_features_a02_features_or_disk_count',
        't01_smooth_or_features_a03_star_or_artifact_count',
        't03_bar_a06_bar_count',
        't03_bar_a07_no_bar_count',
        'id',
        'ra',
        'dec',
        'png_loc',
        'png_ready',
        'sample'
    ]
    unshuffled_catalog = pd.read_csv(catalog_loc, usecols=usecols)
    catalog = shuffle(unshuffled_catalog)
    catalog = catalog[catalog['sample'] == 'original']
    # make label and total_votes columns consistent with decals
    catalog = catalog.rename(index=str, columns={
        't01_smooth_or_features_a01_smooth_count': 'smooth-or-featured_smooth',
        't01_smooth_or_features_a02_features_or_disk_count': 'smooth-or-featured_featured-or-disk',
        't01_smooth_or_features_a03_star_or_artifact_count': 'smooth-or-featured_artifact',
        't03_bar_a06_bar_count': 'bar_yes',  # note that these won't match because DECALS uses strong/weak/none
        't03_bar_a07_no_bar_count': 'bar_no'  # while GZ2 used yes/no
        }
    )
    catalog['smooth-or-featured_total-votes'] = catalog['smooth-or-featured_smooth'] + catalog['smooth-or-featured_featured-or-disk'] + catalog['smooth-or-featured_artifact']
    catalog['bar_total-votes'] = catalog['bar_yes'] + catalog['bar_no']
    catalog['id_str'] = catalog['id'].astype(str)
    catalog.to_csv(save_loc, index=False)


def shuffle(df):
    # THIS IS CRUCIAL. GZ catalog is not properly shuffled, and featured-ness changes systematically
    return df.sample(len(df)).reset_index()

if __name__ == '__main__':
    # assume run from repo root

    # local only, upload the results with dvc. Agnostic of which question to answer
    tweak_joint_catalog(
        catalog_loc='/Volumes/alpha/galaxy_zoo/decals/catalogs/dr5_nsa_v1_0_0_to_upload.fits',
        save_loc='data/decals/joint_catalog_selected_cols.csv'
    )
    tweak_previous_decals_classifications(
        catalog_loc='/data/repos/gzreduction/data/predictions/example_panoptes_predictions.csv',  # will change this TODO
        save_loc='data/decals/previous_classifications_renamed.csv'
    )
    tweak_previous_gz2_classifications(
        catalog_loc='data/gz2/gz2_classifications_and_subjects.csv',
        save_loc='data/gz2/previous_classifications_renamed.csv'
    )
    # remember to add to dvc and push to s3

    # later, run finalise_catalog to apply filters and specify the question to solve
    # this is considered part of the shards, and results are saved to the shards directory
