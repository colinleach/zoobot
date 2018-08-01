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
    df = pd.read_csv(df_loc, usecols=columns_to_save + ['fits_loc', 'png_loc', 'png_ready'])
    logging.info('Loaded {} catalog galaxies with predictions'.format(len(df)))

    # use the majority vote as a label
    # label_col = 'smooth-or-featured_prediction-encoded'
    # df = df[df[label_col] > 0]  # no artifacts
    # df[label_col] = df[label_col] - 1  # 0 for featured

    label_split_value = 0.4
    # use best split of the data is around smooth vote fraction = 0.4
    label_col = 'label'
    df[label_col] = (df['smooth-or-featured_smooth_fraction'] > label_split_value).astype(int)  # 0 for featured

    df = df[df['smooth-or-featured_total-votes'] > 36]  # >36 votes required, gives low count uncertainty

    for size in [128]:  # TODO 96, 256, 424 in 0.4 and 0.5

        train_loc = 'zoobot/data/panoptes_featured_s{}_l{}_train.tfrecord'.format(size, str(label_split_value)[:3])
        test_loc = 'zoobot/data/panoptes_featured_s{}_l{}_test.tfrecord'.format(size, str(label_split_value)[:3])

        catalog_to_tfrecord.write_catalog_to_train_test_tfrecords(
            df,
            label_col,
            train_loc,
            test_loc,
            size,
            columns_to_save)


if __name__ == '__main__':
    save_panoptes_to_tfrecord()
