import logging

import pandas as pd
from zoobot.tfrecord import catalog_to_tfrecord


def save_panoptes_to_tfrecord():

    columns_to_save = [
        'smooth-or-featured_smooth',
        'smooth-or-featured_featured-or-disk',
        'smooth-or-featured_artifact',
        'smooth-or-featured_total-votes',
        'smooth-or-featured_smooth_fraction',
        'smooth-or-featured_featured-or-disk_fraction',
        'smooth-or-featured_artifact_fraction',
        'smooth-or-featured_smooth_min',
        'smooth-or-featured_smooth_max',
        'smooth-or-featured_featured-or-disk_min',
        'smooth-or-featured_featured-or-disk_max',
        'smooth-or-featured_artifact_min',
        'smooth-or-featured_artifact_max',
        'smooth-or-featured_prediction-encoded',  # 0 for artifact, 1 for featured, 2 for smooth
        'classifications_count',
        # 'iauname', string features not yet supported
        'subject_id',
        'nsa_id',
        'ra',
        'dec']

    df_loc = '/data/repos/galaxy-zoo-panoptes/reduction/data/output/panoptes_predictions_with_catalog.csv'
    df = pd.read_csv(df_loc, usecols=columns_to_save + ['fits_loc', 'png_loc', 'png_ready'], nrows=None)
    logging.info('Loaded {} catalog galaxies with predictions'.format(len(df)))

    # use the majority vote as a label
    # label_col = 'smooth-or-featured_prediction-encoded'
    # df = df[df[label_col] > 0]  # no artifacts
    # df[label_col] = df[label_col] - 1  # 0 for featured

    # use best split of the data as a label: here, around smooth vote fraction = 0.4
    label_col = 'label'
    df[label_col] = (df['smooth-or-featured_smooth_fraction'] > 0.4).astype(int)  # 0 for featured

    df = df[df['smooth-or-featured_total-votes'] == 40]  # 40 votes required, gives low count uncertainty

    for size in [28, 64, 128]:

        train_loc = 'data/panoptes_calibration_featured_{}_train.tfrecord'.format(size)
        test_loc = 'data/panoptes_calibration_featured_{}_test.tfrecord'.format(size)

        catalog_to_tfrecord.write_catalog_to_train_test_tfrecords(
            df,
            label_col,
            train_loc,
            test_loc,
            size,
            columns_to_save)


if __name__ == '__main__':
    save_panoptes_to_tfrecord()
