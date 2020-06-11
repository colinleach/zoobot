# load the decals joint catalog and save only the important columns
import logging
import os
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
from astropy.table import Table

from zoobot.science_logic import define_experiment


def create_decals_master_catalog(catalog_loc, classifications_loc, save_loc):
    """Convert zooniverse/decals joint catalog (from decals repo) for active learning and join to previous classifications
    
    Args:
        catalog_loc (str): Load science catalog of galaxies from here.
        classifications_loc (str): Load GZ classifications (see gz-panoptes-reduction repo) from here
        save_loc (str): Save master catalog here
    """
    catalog = pd.read_csv(catalog_loc)
    # conversion messes up strings into bytes
    # for str_col in ['iauname', 'png_loc', 'fits_loc']:
    #     catalog[str_col] = catalog[str_col].apply(lambda x: x.decode('utf-8'))

    # rename columns for convenience (could move this to DECALS repo)
    catalog['nsa_version'] = 'v1_0_0'

    # may need to change png prefix
    catalog = catalog.rename(index=str, columns={
        'fits_loc': 'local_fits_loc',
        'png_loc': 'local_png_loc',
        'z': 'redshift'
    })

    print('Galaxies: {}'.format(len(catalog)))
    # tweak png locs according to current machine
    catalog = specify_file_locs(catalog, 'decals')

    classifications = pd.read_csv(classifications_loc)
    # add all historical classifications (before starting any active learning)
    df = pd.merge(catalog, classifications, how='left',
                  on='iauname')  # many rows will have None   
    df['id_str'] = df['iauname'] 
    df = define_experiment.drop_duplicates(df)
    df.to_csv(save_loc, index=False)


def create_gz2_master_catalog(catalog_loc: str, save_loc: str):
    usecols = [
        't01_smooth_or_features_a01_smooth_count',
        't01_smooth_or_features_a02_features_or_disk_count',
        't01_smooth_or_features_a03_star_or_artifact_count',
        't02_edgeon_a04_yes_count',
        't02_edgeon_a05_no_count',
        't03_bar_a06_bar_count',
        't03_bar_a07_no_bar_count',
        't04_spiral_a08_spiral_count',
        't04_spiral_a09_no_spiral_count',
        't05_bulge_prominence_a10_no_bulge_count',
        't05_bulge_prominence_a11_just_noticeable_count',
        't05_bulge_prominence_a12_obvious_count',
        't05_bulge_prominence_a13_dominant_count',
        't06_odd_a14_yes_count',
        't06_odd_a15_no_count',
        't07_rounded_a16_completely_round_count',
        't07_rounded_a17_in_between_count',
        't07_rounded_a18_cigar_shaped_count',
        # skip t08, what is odd, as multiple choice
        't09_bulge_shape_a25_rounded_count',
        't09_bulge_shape_a26_boxy_count',
        't09_bulge_shape_a27_no_bulge_count',
        't10_arms_winding_a28_tight_count',
        't10_arms_winding_a29_medium_count',
        't10_arms_winding_a30_loose_count',
        't11_arms_number_a31_1_count',
        't11_arms_number_a32_2_count',
        't11_arms_number_a33_3_count',
        't11_arms_number_a34_4_count',
        't11_arms_number_a36_more_than_4_count',
        't11_arms_number_a37_cant_tell_count',
        'dr7objid',
        'ra_subject',
        'dec_subject',
        'png_loc',
        'png_ready',
        'sample'
    ]
    unshuffled_catalog = pd.read_csv(catalog_loc, usecols=usecols)
    df = shuffle(unshuffled_catalog)
    df = df[df['sample'] == 'original']
    # make label and total_votes columns consistent with decals
    df = df.rename(index=str, columns={
        'ra_subject': 'ra',
        'dec_subject': 'dec',
        'png_loc': 'local_png_loc',  # though I will overwrite in a mo anyway
        't01_smooth_or_features_a01_smooth_count': 'smooth-or-featured_smooth',
        't01_smooth_or_features_a02_features_or_disk_count': 'smooth-or-featured_featured-or-disk',
        't01_smooth_or_features_a03_star_or_artifact_count': 'smooth-or-featured_artifact',
        't02_edgeon_a04_yes_count': 'disk-edge-on_yes',
        't02_edgeon_a05_no_count': 'disk-edge-on_no',
        #  note that these won't match because DECALS uses strong/weak/none
        't03_bar_a06_bar_count': 'bar_yes',
        't03_bar_a07_no_bar_count': 'bar_no',  # while GZ2 used yes/no
        't04_spiral_a08_spiral_count': 'has-spiral-arms_yes',
        't04_spiral_a09_no_spiral_count': 'has-spiral-arms_no',
        # similarly won't match decals bulge sizes
        't05_bulge_prominence_a10_no_bulge_count': 'bulge-size_no',
        't05_bulge_prominence_a11_just_noticeable_count': 'bulge-size_just-noticeable',
        't05_bulge_prominence_a12_obvious_count': 'bulge-size_obvious',
        't05_bulge_prominence_a13_dominant_count': 'bulge-size_dominant',
        't06_odd_a14_yes_count': 'something-odd_yes',
        't06_odd_a15_no_count': 'something-odd_no',
        't07_rounded_a16_completely_round_count': 'how-rounded_round',
        't07_rounded_a17_in_between_count': 'how-rounded_in-between',
        't07_rounded_a18_cigar_shaped_count': 'how-rounded_cigar',
        # skip t08, what is odd, as multiple choice
        't09_bulge_shape_a25_rounded_count': 'bulge-shape_round',
        't09_bulge_shape_a26_boxy_count': 'bulge-shape_boxy',
        't09_bulge_shape_a27_no_bulge_count': 'bulge-shape_no-bulge',
        't10_arms_winding_a28_tight_count': 'spiral-winding_tight',
        't10_arms_winding_a29_medium_count': 'spiral-winding_medium',
        't10_arms_winding_a30_loose_count': 'spiral-winding_loose',
        't11_arms_number_a31_1_count': 'spiral-count_1',
        't11_arms_number_a32_2_count': 'spiral-count_2',
        't11_arms_number_a33_3_count': 'spiral-count_3',
        't11_arms_number_a34_4_count': 'spiral-count_4',
        't11_arms_number_a36_more_than_4_count': 'spiral-count_more-than-4',
        't11_arms_number_a37_cant_tell_count': 'spiral-count_cant-tell'
    })

    df['smooth-or-featured_total-votes'] = df['smooth-or-featured_smooth'] + \
        df['smooth-or-featured_featured-or-disk'] + \
        df['smooth-or-featured_artifact']
    df['disk-edge-on_total-votes'] = df['disk-edge-on_yes'] + df['disk-edge-on_no']
    df['bar_total-votes'] = df['bar_yes'] + df['bar_no']
    df['has-spiral-arms_total-votes'] = df['has-spiral-arms_yes'] + df['has-spiral-arms_no']
    df['bulge-size_total-votes'] = df['bulge-size_no'] + df['bulge-size_just-noticeable'] + df['bulge-size_obvious'] + df['bulge-size_dominant']
    df['something-odd_total-votes'] = df['something-odd_yes'] + df['something-odd_no']
    df['how-rounded_total-votes'] = df['how-rounded_round'] + df['how-rounded_in-between'] + df['how-rounded_cigar']
    df['bulge-shape_total-votes'] = df['bulge-shape_round'] + df['bulge-shape_boxy'] + df['bulge-shape_no-bulge']
    df['spiral-winding_total-votes'] = df['spiral-winding_tight'] + df['spiral-winding_medium'] + df['spiral-winding_loose']
    df['spiral-count_total-votes'] = df['spiral-count_1'] + df['spiral-count_2'] + df['spiral-count_3'] + df['spiral-count_4'] + df['spiral-count_more-than-4'] + df['spiral-count_cant-tell']

    df['id_str'] = df['dr7objid'].apply(lambda x: 'dr7objid_' + str(x))  # useful to describe, and  avoids automatic string->int conversion problems
    # change to be inside data folder, specified relative to repo root. Use local_png_loc (later) for absolute path
    df['png_loc'] = df['local_png_loc'].apply(lambda x: x.replace('/Volumes/alpha/', '').replace('gz2/', '').replace('decals/', ''))
    print(df['png_loc'])
    df = specify_file_locs(df, 'gz2')
    df.to_csv(save_loc, index=False)


def specify_file_locs(df, target):
    """
    Add 'file_loc' which points to pngs at expected absolute EC2 path
    Remove 'png_loc (png relative to repo root) to avoid confusion
    """
    png_root_loc =  get_png_root_loc(target)
    # change to be inside data folder, specified relative to repo root
    df['local_png_loc'] = df['png_loc'].apply(
        lambda x: os.path.join(png_root_loc, x)
    )

    df['file_loc'] = df['local_png_loc']
    assert all(loc for loc in df['file_loc'])
    del df['png_loc']  # else may load this by default
    print(df['file_loc'].sample(5))
    check_no_missing_files(df['file_loc'], max_to_check=1000)
    return df


def get_png_root_loc(target):
    # EC2
    if os.path.isdir('/home/ec2-user'):
        return f'/home/ec2-user/root/repos/zoobot/data/{target}'
    # laptop
    elif os.path.isdir('/home/walml'):
        return f'/media/walml/beta/galaxy_zoo/{target}/'
    # EC2 Ubuntu
    elif os.path.isdir('/home/ubuntu'):
        return f'/home/ubuntu/root/repos/zoobot/data/{target}'
    # Oxford Desktop
    elif os.path.isdir('/data/repos'):
        # logging.critical('Local master catalog - do not use on EC2!')
        return f'/Volumes/alpha/{target}'
    # ARC
    elif os.path.isdir('/data/phys-zooniverse/chri5177'):
        return f'/data/phys-zooniverse/chri5177/{target}'
    else:
        raise ValueError('Cannot work out appropriate png root')


def check_no_missing_files(locs, max_to_check=None):
    # locs_missing = [not os.path.isfile(path) for path in tqdm(locs)]
    # if any(locs_missing):
        # raise ValueError('Missing {} files e.g. {}'.format(
        # np.sum(locs_missing), locs[locs_missing][0]))
    print('Checking no missing files')
    if max_to_check is not None:
        if len(locs) > max_to_check:
            locs = np.random.choice(locs, max_to_check)
    for loc in tqdm(locs):
        if not os.path.isfile(loc):
            raise ValueError('Missing ' + loc)




def shuffle(df):
    # THIS IS CRUCIAL. GZ catalog is not properly shuffled, and featured-ness changes systematically
    return df.sample(len(df)).reset_index()


if __name__ == '__main__':

    """
    Decals:
        python zoobot/science_logic/prepare_catalogs.py /media/walml/beta/decals/catalogs/decals_dr5_uploadable_master_catalog_nov_2019.csv /media/walml/beta/decals/results/classifications_oct_3_2019.csv data/decals/decals_master_catalog.csv

    GZ2:
        python zoobot/science_logic/prepare_catalogs.py /media/walml/beta/galaxy_zoo/gz2/subjects/gz2_classifications_and_subjects.csv '' data/gz2/gz2_master_catalog.csv
        python zoobot/science_logic/prepare_catalogs.py $DATA/repos/zoobot/data/gz2/gz2_classifications_and_subjects.csv '' data/gz2/gz2_master_catalog_arc.csv
    """
    parser = argparse.ArgumentParser(description='Make shards')
    parser.add_argument('catalog_loc', type=str,
                        help='Path to csv of decals catalog (dr5 only), from decals repo')
    parser.add_argument('classifications_loc', type=str,
                        help='Latest streamed classifications, from gzreduction')
    parser.add_argument('save_loc', type=str,
                        help='Place active learning master catalog here')
    args = parser.parse_args()
    # assume run from repo root
    # LOCAL ONLY upload the results with dvc.

    # should run full reduction first and place in classifications_loc
    # see mwalmsley/gzreduction/get_latest.py

    if 'gz2' in args.catalog_loc:
        create_gz2_master_catalog(
            catalog_loc=args.catalog_loc,
            save_loc=args.save_loc
        )
    else:
        create_decals_master_catalog(
            catalog_loc=args.catalog_loc,
            classifications_loc=args.classifications_loc,
            save_loc=args.save_loc
        )

    # remember to add to dvc and push to s3

    # Agnostic of which question to answer
    # later, run finalise_catalog to apply filters and specify the question to solve
    # this is considered part of the shards, and results are saved to the shards directory

    # df = pd.read_csv('data/decals/decals_master_catalog.csv')
    # df['file_loc'] = df['file_loc'].apply(lambda x: '/home/ubuntu' + x)
    # print(df['file_loc'][0])
    # df.to_csv('data/decals/decals_master_catalog.csv', index=False)
