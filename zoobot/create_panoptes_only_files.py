"""Temporary script to take only fits files with panoptes labels, and put them in a separate folder
Useful to prototype maching learning
Adds column with fits loc relative to an unknown root
"""
import os
from subprocess import call

import pandas as pd

from zoobot.tests import TEST_EXAMPLE_DIR

df = pd.read_csv(
    os.path.join(TEST_EXAMPLE_DIR, 'panoptes_mock_predictions.csv'), 
    dtype={'id_loc': str, 'label': float},
    # nrows=10
)

current_fits_native_dir = '/Volumes/alpha/decals/fits_native'
new_fits_native_dir = '/data/aws/s3/galaxy-zoo/decals/fits_native'

df['fits_loc_old'] = df['fits_loc']


current_dir_chars = len(current_fits_native_dir)
df['fits_loc_relative'] = df['fits_loc_old'].apply(lambda x: x[current_dir_chars:])
df['fits_loc'] = new_fits_native_dir + df['fits_loc_relative']

print(df.iloc[0]['fits_loc_old'])
print(df.iloc[0]['fits_loc_relative'])
print(df.iloc[0]['fits_loc'])

for _, row in df.iterrows():
    # copy file to new native directory
    target_dir = os.path.dirname(row['fits_loc'])
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    call(["cp", row['fits_loc_old'], row['fits_loc']])


del df['fits_loc_old']

df.to_csv('panoptes_predictions.csv', index=False)
