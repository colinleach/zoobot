# load the decals joint catalog and save only the important columns
import logging

import pandas as pd
from astropy.table import Table

def tweak_joint_catalog():
    # rename and resave a few joint catalog columns, to make uploading easier
    joint_catalog_table = Table.read('/Volumes/alpha/galaxy_zoo/decals/catalogs/dr5_nsa_v1_0_0_to_upload.fits')
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
    df.to_csv('data/decals/joint_catalog_selected_cols.csv', index=False)

# def tweak_previous_classifications():
#     # create a labelled catalog to use when starting all iterations
#     # this will be used to make the first training shard
#     df = pd.read_csv('/data/repos/gzreduction/data/predictions/example_panoptes-predictions.csv')
    # should actually be good to go!




if __name__ == '__main__':
    tweak_joint_catalog()