import logging
import os
import argparse

import pandas as pd
from zoobot.tfrecord import catalog_to_tfrecord
from zoobot import settings

def save_panoptes_to_tfrecord(catalog_loc, tfrecord_dir):

    useful_cols = [
        'smooth-or-featured_total-votes',
        'smooth-or-featured_smooth_fraction',
        'classifications_count',
        'id_str',
        'nsa_id',
        'ra',
        'dec']

    df = pd.read_csv(catalog_loc, usecols=useful_cols + ['fits_loc', 'png_loc', 'png_ready'])
    df['id_str'] = df['subject_id'].astype(str)

    logging.info('Loaded {} catalog galaxies with predictions'.format(len(df)))

    # use the majority vote as a label
    # label_col = 'smooth-or-featured_prediction-encoded'
    # df = df[df[label_col] > 0]  # no artifacts
    # df[label_col] = df[label_col] - 1  # 0 for featured

    # label_split_value = 0.4
    # use best split of the data is around smooth vote fraction = 0.4

    # df[label_col] = (df['smooth-or-featured_smooth_fraction'] > label_split_value).astype(int)  # 0 for featured
    label_col = 'label'
    df[label_col] = df['smooth-or-featured_smooth_fraction']  # 0 for featured

    df = df[df['smooth-or-featured_total-votes'] > 36]  # >36 votes required, gives low count uncertainty

    for size in [128]:  # 96, 128, 256, 424 

        # train_loc = os.path.join(tfrecord_dir, 'panoptes_featured_s{}_l{}_train.tfrecord'.format(size, str(label_split_value)[:3]))
        # test_loc = os.path.join(tfrecord_dir, 'panoptes_featured_s{}_l{}_test.tfrecord'.format(size, str(label_split_value)[:3]))
        train_loc = os.path.join(tfrecord_dir, 'panoptes_featured_s{}_lfloat_train.tfrecord'.format(size))
        test_loc = os.path.join(tfrecord_dir, 'panoptes_featured_s{}_lfloat_test.tfrecord'.format(size))

        catalog_to_tfrecord.write_catalog_to_train_test_tfrecords(
            df,
            train_loc,
            test_loc,
            size,
            useful_cols + [label_col])

if __name__ == '__main__':

    save_panoptes_to_tfrecord('data/panoptes_predictions_selected.csv', 'data/basic_split')
