import os

import pandas as pd
from zoobot.tfrecord import catalog_to_tfrecord


def save_gz2_to_tfrecord(catalog_loc, save_dir):

    assert os.path.isfile(catalog_loc)
    assert os.path.isdir(save_dir)

    for size in [128]:

        # needs update
        columns_to_save = [
            't01_smooth_or_features_a01_smooth_count',
            't01_smooth_or_features_a01_smooth_weighted_fraction',  # annoyingly, I only saved the weighted fractions. Should be quite similar, I hope.
            't01_smooth_or_features_a02_features_or_disk_count',
            't01_smooth_or_features_a03_star_or_artifact_count',
            'id',
            'ra',
            'dec'
        ]

        # only exists if zoobot/get_catalogs/gz2 instructions have been followed
        df = pd.read_csv(catalog_loc,
                         usecols=columns_to_save + ['png_loc', 'png_ready'],
                         nrows=None)

        # previous catalog didn't include total classifications/votes, so we'll need to work around that for now
        df['smooth-or-featured_total-votes'] = df['t01_smooth_or_features_a01_smooth_count'] + df['t01_smooth_or_features_a02_features_or_disk_count'] + df['t01_smooth_or_features_a03_star_or_artifact_count']
        df = df[df['smooth-or-featured_total-votes'] > 36]  # >36 votes required, gives low count uncertainty

        # for consistency
        df['id_str'] = df['id'].astype(str)

        train_loc = os.path.join(save_dir, 'gz2_smooth_frac_{}_train.tfrecord'.format(size))
        test_loc = os.path.join(save_dir, 'gz2_smooth_frac_{}_test.tfrecord'.format(size))

        label_col = 'label'
        df[label_col] = df['t01_smooth_or_features_a01_smooth_weighted_fraction']

        catalog_to_tfrecord.write_catalog_to_train_test_tfrecords(
            df=df,
            train_loc=train_loc,
            test_loc=test_loc,
            img_size=size,
            columns_to_save=columns_to_save + [label_col, 'id_str'],
            source='png')


if __name__ == '__main__':
    catalog_loc = '/data/galaxy_zoo/gz2/catalogs/basic_regression_labels_downloaded.csv'
    save_dir = 'data/basic_split_gz2'

    # print(pd.read_csv(catalog_loc, nrows=5).columns.values)
    # exit(0)

    save_gz2_to_tfrecord(catalog_loc, save_dir)