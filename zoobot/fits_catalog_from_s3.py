
# given a catalog with fits_loc_relative, and target dir
# work out the s3 location
# copy to target dir
# save catalog with new accurate fits loc

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

from zoobot.tests import TEST_EXAMPLE_DIR


df = pd.read_csv(
    os.path.join(TEST_EXAMPLE_DIR, 'panoptes_predictions.csv'), 
    dtype={'id_loc': str, 'label': float},
    nrows=10
)

s3_dir = 's3://galaxy-zoo/decals/fits_native'
df['fits_loc_s3'] = s3_dir + df['fits_loc_relative']

# target_dir = '/users/mikewalmsley/pretend_ec2_root/fits_native'
target_dir = '/home/ec2-user/fits_native'
if not os.path.isdir(target_dir):
    os.mkdir(target_dir)
df['fits_loc'] = target_dir + df['fits_loc_relative']

df.to_csv('panoptes_predictions.csv', index=False)

for _, row in df.iterrows():
    # copy file to new native directory from s3
    # final_dir = os.path.dirname(row['fits_loc'])
    # penultimate_dir = os.path.dirname(final_dir)
    # if not os.path.isdir(penultimate_dir):
    #     os.mkdir(penultimate_dir)
    # if not os.path.isdir(final_dir):
    #     os.mkdir(final_dir)
    # can copy one at a time, but much much slower
    # call(["aws", "s3", "cp", row['fits_loc_s3'], row['fits_loc']])
        
    # better to have all in one bucket and sync
    # can copy into this bucket beforehand, perhaps?
    call(["aws", "s3", "sync", 's3://galaxy-zoo/decals/fits_native', target_dir])


# del df['fits_loc_old']