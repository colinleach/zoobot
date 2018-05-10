import logging
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from astropy.io import fits
from tqdm import tqdm

from zoobot.tfrecord import create_tfrecord, image_utils


def write_catalog_to_train_test_tfrecords(df, label_col, train_loc, test_loc, img_size, columns_to_save, train_test_fraction=0.8):
    train_test_split = int(train_test_fraction * len(df))

    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df[:train_test_split].copy()
    test_df = df[train_test_split:].copy()

    assert not train_df.empty
    assert not test_df.empty

    write_image_df_to_tfrecord(train_df, label_col, train_loc, img_size, columns_to_save, append=False, source='fits')
    write_image_df_to_tfrecord(test_df, label_col, test_loc, img_size, columns_to_save, append=False, source='fits')

    return train_df, test_df


def write_image_df_to_tfrecord(df, label_col, tfrecord_loc, img_size, columns_to_save, append=False, source='fits'):

    if not append:
        if os.path.exists(tfrecord_loc):
            print('{} already exists - deleting'.format(tfrecord_loc))
            os.remove(tfrecord_loc)

    writer = tf.python_io.TFRecordWriter(tfrecord_loc)

    dimensions = img_size, img_size

    for subject_n, subject in tqdm(df.iterrows(), total=len(df), unit=' subjects saved'):
        if subject['png_ready']:
            if source == 'fits':
                pil_img = load_fits_as_pil(subject)
            elif source == 'png':
                pil_img = load_png_as_pil(subject)
            else:
                logging.critical('Fatal error: image source "{}" not understood'.format(source))
                raise ValueError

            # to align with north/east TODO refactor this to make sure it matches downloader?
            final_pil_img = pil_img.resize(size=(img_size, img_size), resample=Image.LANCZOS).transpose(Image.FLIP_TOP_BOTTOM)

            matrix = np.array(final_pil_img)

            label = int(subject[label_col])
            extra_data = {}
            for col in columns_to_save:
                extra_data.update({col: subject[col]})

            create_tfrecord.image_to_tfrecord(matrix, label, writer, extra_data)

    writer.close()
    sys.stdout.flush()


def load_png_as_pil(subject):
    return Image.open(subject['png_loc'])


def load_fits_as_pil(subject):  # TODO refactor to make sure this aligns with downloader
    img = fits.getdata(subject['fits_loc'])

    _scales = dict(
        g=(2, 0.008),
        r=(1, 0.014),
        z=(0, 0.019))

    _mnmx = (-0.5, 300)

    rgb_img = image_utils.dr2_style_rgb(
        (img[0, :, :], img[1, :, :], img[2, :, :]),
        'grz',
        mnmx=_mnmx,
        arcsinh=1.,
        scales=_scales,
        desaturate=True)
    return Image.fromarray(np.uint8(rgb_img * 255.), mode='RGB')
