"""
Temporary script to take only fits files with panoptes labels, and put them in a separate folder
Useful to prototype maching learning
Adds column with fits loc relative to an unknown root, and optionally saves to new file

Initial catalog comes from `panoptes_mock_predictions.csv`, on s3://mikewalmsley
New catalog (i.e. with updated `fits_loc`, `fits_loc_relative`) optionally saved
"""
import os
from subprocess import call

import pandas as pd
from tqdm import tqdm

from zoobot.tests import TEST_EXAMPLE_DIR

if __name__ == '__main__':

    desktop = True
    laptop = False

    # df_loc = os.path.join(TEST_EXAMPLE_DIR, 'panoptes_mock_predictions.csv')
    df_loc = '/data/repos/zoobot/data/2018-11-05_panoptes_predictions_with_catalog.csv'
    assert os.path.exists(df_loc)

    df = pd.read_csv(
        df_loc, 
        dtype={'id_str': str, 'label': float},
        # nrows=10
    )

    current_fits_native_dir = '/Volumes/alpha/decals/fits_native'


    if desktop:
        assert not laptop
        new_fits_native_dir = '/data/repos/zoobot/data/fits_native'  

    if laptop:
        assert not desktop
        new_fits_native_dir = '/Users/mikewalmsley/aws/s3/galaxy-zoo/decals/fits_native'


    df['fits_loc_old'] = df['fits_loc']


    current_dir_chars = len(current_fits_native_dir)
    df['fits_loc_relative'] = df['fits_loc_old'].apply(lambda x: x[current_dir_chars:])
    df['fits_loc'] = new_fits_native_dir + df['fits_loc_relative']

    print(df.iloc[0]['fits_loc_old'])
    print(df.iloc[0]['fits_loc_relative'])
    print(df.iloc[0]['fits_loc'])

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # copy file to new native directory
        target_dir = os.path.dirname(row['fits_loc'])
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        assert row['fits_loc_old']
        call(["cp", row['fits_loc_old'], row['fits_loc']])


    del df['fits_loc_old']

    df.to_csv('/data/repos/zoobot/data/panoptes_predictions_selected.csv', index=False)
