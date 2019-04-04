# load the decals joint catalog and save only the important columns
import logging
import os

import pandas as pd
from astropy.table import Table


# def create_decals_master_catalog(catalog_loc, classifications_loc, save_loc):

#     catalog = Table.read(catalog_loc).to_pandas()
#     # conversion messes up strings into bytes
#     for str_col in ['iauname', 'png_loc', 'fits_loc']:
#         catalog[str_col] = catalog[str_col].apply(lambda x: x.decode('utf-8'))

#     catalog['nsa_version'] = 'v1_0_0'
#     catalog['id_str'] = catalog['iauname']  # not the subject id! We want unique per galaxy, not per subject
#     catalog = catalog.rename(index=str, columns={
#         'fits_loc': 'local_fits_loc',
#         'png_loc': 'local_png_loc',
#         'z': 'redshift'
#     })

#     catalog['png_loc'] = df['local_png_loc'].apply(lambda x: 'data/' + x.lstrip('/Volumes/alpha'))  # change to be inside data folder, specified relative to repo root
#     print(df.iloc[0]['png_loc'])
#     print('Galaxies: {}'.format(len(df)))
#     catalog = specify_file_locs(catalog)
    
#     classifications = pd.read_csv(classifications_loc)
#     # create a labelled catalog to use when starting all iterations
#     # this will be used to make the first training shard
#     # catalog['id_str'] = catalog['subject_id'].astype(str)
#     # catalog['id_str'] = catalog['iauname']
#     df = pd.merge(catalog, classifications, how='inner', on=TODO)
#     # r
#     assert not any(df['id_str'].duplicated())
#     return df


def create_gz2_master_catalog(catalog_loc, save_loc):
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
    df = shuffle(unshuffled_catalog)
    df = df[df['sample'] == 'original']
    # make label and total_votes columns consistent with decals
    df = df.rename(index=str, columns={
        't01_smooth_or_features_a01_smooth_count': 'smooth-or-featured_smooth',
        't01_smooth_or_features_a02_features_or_disk_count': 'smooth-or-featured_featured-or-disk',
        't01_smooth_or_features_a03_star_or_artifact_count': 'smooth-or-featured_artifact',
        't03_bar_a06_bar_count': 'bar_yes',  # note that these won't match because DECALS uses strong/weak/none
        't03_bar_a07_no_bar_count': 'bar_no',  # while GZ2 used yes/no
        'png_loc': 'local_png_loc'  # absolute file loc on local desktop
        }
    )
    df['smooth-or-featured_total-votes'] = df['smooth-or-featured_smooth'] + df['smooth-or-featured_featured-or-disk'] + df['smooth-or-featured_artifact']
    df['bar_total-votes'] = df['bar_yes'] + df['bar_no']
    df['id_str'] = df['id'].astype(str)
    df['png_loc'] = df['local_png_loc'].apply(lambda x: 'data/' + x.lstrip('/Volumes/alpha'))  # change to be inside data folder, specified relative to repo root
    df = specify_file_locs(df)  # expected absolute file loc on EC2
    df.to_csv(save_loc, index=False)


def specify_file_locs(df):
    """
    Add 'file_loc' which points to pngs at expected absolute EC2 path
    Remove 'png_loc (png relative to repo root) to avoid confusion
    """
    df['file_loc'] = '/root/repos/zoobot/' + df['png_loc']  # now expects this to point to png loc relative to repo root
    assert all(loc for loc in df['file_loc'])
    del df['png_loc']  # else may load this by default
    print(df['file_loc'].sample(5))
    return df


def shuffle(df):
    # THIS IS CRUCIAL. GZ catalog is not properly shuffled, and featured-ness changes systematically
    return df.sample(len(df)).reset_index()


if __name__ == '__main__':
    # assume run from repo root

    # local only, upload the results with dvc. Agnostic of which question to answer
    # tweak_joint_catalog(
    #     catalog_loc='/Volumes/alpha/galaxy_zoo/decals/catalogs/dr5_nsa_v1_0_0_to_upload.fits',
    #     save_loc='data/decals/joint_catalog_selected_cols.csv'
    # )
    # tweak_previous_decals_classifications(
    #     catalog_loc='/data/repos/gzreduction/data/predictions/example_panoptes_predictions.csv',  # will change this TODO
    #     save_loc='data/decals/previous_classifications_renamed.csv'
    # )
    create_gz2_master_catalog(
        catalog_loc='data/gz2/gz2_classifications_and_subjects.csv',
        save_loc='data/gz2/gz2_master_catalog.csv'
    )
    # remember to add to dvc and push to s3

    # later, run finalise_catalog to apply filters and specify the question to solve
    # this is considered part of the shards, and results are saved to the shards directory
