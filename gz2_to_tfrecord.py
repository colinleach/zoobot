import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf

from create_tfrecord import image_to_tfrecord


def write_image_df_to_tfrecord(df, tfrecord_loc, img_size):

    if os.path.exists(tfrecord_loc):
        print('{} already exists - deleting'.format(tfrecord_loc))
        os.remove(tfrecord_loc)

    writer = tf.python_io.TFRecordWriter(tfrecord_loc)

    dimensions = size, size

    for subject_n, subject in tqdm(df.iterrows(), total=len(df), unit=' subjects saved'):
        # print(subject_n)
        if subject['png_ready']:
            img = Image.open(subject['png_loc'])
            img.thumbnail(dimensions)  # inplace on img
            matrix = np.array(img)
            label = int(subject['t04_spiral_a08_spiral_weighted_fraction'])
            extra_data = {}
            for col in columns_to_save:
                extra_data.update({col: subject[col]})

            image_to_tfrecord(matrix, label, writer, extra_data)

    writer.close()
    sys.stdout.flush()


if __name__ == '__main__':

    for size in [128, 256, 512]:

        train_test_fraction = 0.8

        columns_to_save = ['t04_spiral_a08_spiral_count',
                           't04_spiral_a09_no_spiral_count',
                           't04_spiral_a08_spiral_weighted_fraction',
                           'id',
                           'ra',
                           'dec']

        df = pd.read_csv('/data/galaxy_zoo/gz2/subjects/all_labels_downloaded.csv',
                         usecols=columns_to_save + ['png_loc', 'png_ready'],
                         nrows=None)

        train_test_split = int(0.8*len(df))

        df = df.sample(frac=1).reset_index(drop=True)
        train_df = df[:train_test_split].copy()
        test_df = df[train_test_split:].copy()

        print(len(train_df))
        print(len(test_df))

        train_tfrecord_loc = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_train.tfrecord'.format(size)
        test_tfrecord_loc = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}_test.tfrecord'.format(size)

        write_image_df_to_tfrecord(train_df, train_tfrecord_loc, size)
        write_image_df_to_tfrecord(test_df, test_tfrecord_loc, size)

        assert os.path.exists(train_tfrecord_loc)
        assert os.path.exists(test_tfrecord_loc)
