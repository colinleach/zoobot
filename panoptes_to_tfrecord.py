import logging
import os
import argparse

import pandas as pd
from zoobot.tfrecord import catalog_to_tfrecord
from zoobot import settings

def save_panoptes_to_tfrecord(catalog_loc, tfrecord_dir):

    to_save = [
        'smooth-or-featured_total-votes',
        'smooth-or-featured_smooth_fraction',
        'classifications_count',
        'nsa_id',
        'ra',
        'dec']

    df = pd.read_csv(catalog_loc, usecols=to_save + ['fits_loc', 'png_loc', 'png_ready', 'subject_id'])
    df['id_str'] = df['subject_id'].astype(str)

    logging.info('Loaded {} catalog galaxies with predictions'.format(len(df)))

    label_col = 'label'
    df[label_col] = df['smooth-or-featured_smooth_fraction']  # 0 for featured

    df = df[df['smooth-or-featured_total-votes'] > 36]  # >36 votes required, gives low count uncertainty

    for size in [128]:  # 96, 128, 256, 424 
        train_loc = os.path.join(tfrecord_dir, 'panoptes_featured_s{}_lfloat_train.tfrecord'.format(size))
        test_loc = os.path.join(tfrecord_dir, 'panoptes_featured_s{}_lfloat_test.tfrecord'.format(size))

        catalog_to_tfrecord.write_catalog_to_train_test_tfrecords(
            df,
            train_loc,
            test_loc,
            size,
            to_save + [label_col, 'id_str'])

if __name__ == '__main__':

    save_panoptes_to_tfrecord('data/panoptes_predictions_selected.csv', 'data/basic_split')
