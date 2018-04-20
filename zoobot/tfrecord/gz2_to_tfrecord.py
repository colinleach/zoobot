import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from zoobot.tfrecord import create_tfrecord


def write_catalog_to_train_test_tfrecords(df, train_loc, test_loc, img_size, columns_to_save, train_test_fraction=0.8):
    train_test_split = int(train_test_fraction * len(df))

    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df[:train_test_split].copy()
    test_df = df[train_test_split:].copy()

    assert not train_df.empty
    assert not test_df.empty

    write_image_df_to_tfrecord(train_df, train_loc, img_size, columns_to_save)
    write_image_df_to_tfrecord(test_df, test_loc, img_size, columns_to_save)


def write_image_df_to_tfrecord(df, tfrecord_loc, img_size, columns_to_save):

    if os.path.exists(tfrecord_loc):
        print('{} already exists - deleting'.format(tfrecord_loc))
        os.remove(tfrecord_loc)

    writer = tf.python_io.TFRecordWriter(tfrecord_loc)

    dimensions = img_size, img_size

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
            print(extra_data)

            create_tfrecord.image_to_tfrecord(matrix, label, writer, extra_data)

    writer.close()
    sys.stdout.flush()


if __name__ == '__main__':

    for size in [128, 256, 512]:

        columns_to_save = ['t04_spiral_a08_spiral_count',
                           't04_spiral_a09_no_spiral_count',
                           't04_spiral_a08_spiral_weighted_fraction',
                           'id',
                           'ra',
                           'dec']

        df = pd.read_csv('/data/galaxy_zoo/gz2/subjects/all_labels_downloaded.csv',
                         usecols=columns_to_save + ['png_loc', 'png_ready'],
                         nrows=None)

        write_catalog_to_train_test_tfrecords(df, size, columns_to_save)
