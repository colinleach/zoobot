import logging

import pandas as pd
import numpy as np

import matplotlib.colors
from astropy.io import fits
from PIL import Image
from photutils import make_source_mask

from zoobot.tfrecord import catalog_to_tfrecord


def get_image_mask(img):
    return make_source_mask(img, snr=2, npixels=5, dilate_size=11)


def mask_background(img):
    return img * get_image_mask(img)


def clip_img(img):
    source_img = mask_background(img)  # doesn't mask, but uses mask to estimate source statistics
    return np.clip(img, a_min=0, a_max=source_img.mean() + source_img.std() * 8)


def normalise(img):
    return matplotlib.colors.Normalize(vmin=None, vmax=None, clip=False)(img)


def preprocess_img(img):
    return normalise(np.sqrt(clip_img(img)))


def load_illustris_as_pil(subject):  # should be dict with 'fits_loc' component
    img = fits.getdata(subject['fits_loc'])
    return Image.fromarray(preprocess_img(img) * 255)


def split_carefully(df, label_col, train_loc, test_loc, img_size, columns_to_save, train_test_fraction=0.8):
    # TODO this should be refactored into catalog utils for general case
    # Don't split blindly. Make sure each galaxy id is in only one set.
    df = df.sort_values('id')

    train_test_split = int(train_test_fraction * len(df))
    train_df = df[:train_test_split].copy().sample(frac=1).reset_index(drop=True)
    test_df = df[train_test_split:].copy().sample(frac=1).reset_index(drop=True)

    # ensure there are no shared galaxies
    train_df = train_df[~train_df['id'].isin(set(test_df['id']))]
    assert set(train_df['id']).isdisjoint(set(test_df['id']))

    train_df.to_csv(train_loc + '.csv')  # ugly but effective
    test_df.to_csv(test_loc + '.csv')

    assert not train_df.empty
    assert not test_df.empty

    # record df contents
    logging.debug('Train df at {}'.format(train_loc))
    logging.debug(train_df['name'].value_counts())
    logging.debug(train_df['merger'].value_counts())
    logging.debug('Train df at {}'.format(test_loc))
    logging.debug(test_df['name'].value_counts())
    logging.debug(test_df['merger'].value_counts())

    catalog_to_tfrecord.write_image_df_to_tfrecord(train_df, label_col, train_loc, img_size, columns_to_save, append=False, source='fits', load_fits_as_pil=load_illustris_as_pil)
    catalog_to_tfrecord.write_image_df_to_tfrecord(test_df, label_col, test_loc, img_size, columns_to_save, append=False, source='fits', load_fits_as_pil=load_illustris_as_pil)


def remove_known_galaxies_from_random_sample(df):
    # remove any mergers (i.e. already in major/minor merger class) from random_sample - use as merger-free sample
    known_ids = set(df[df['name'] != 'random_sample']['id'])
    random_ids = set(df[df['name'] == 'random_sample']['id'])

    known_in_random_sample = random_ids.intersection(known_ids)

    # galaxy id should not be in both known and random sample, or galaxy name should not be in random_sample
    df = df[(~df['id'].isin(known_in_random_sample)) | (df['name'] != 'random_sample')]

    return df


if __name__ == '__main__':

    logging.basicConfig(
        filename='illustris_to_tfrecord.log',
        format='%(levelname)s:%(message)s',
        filemode='w',
        level=logging.DEBUG)

    catalog_loc = 'catalogs/illustris_galaxies.csv'
    df = pd.read_csv(catalog_loc)

    # random_sample has 320 repeated galaxies (sample with replacement?) - keep only one
    df = df.drop_duplicates(subset=['id', 'name', 'camera'])

    # random_sample includes galaxies already in other categories - remove them from random_sample
    df = remove_known_galaxies_from_random_sample(df)

    assert all(df['id'].value_counts() == 4)  # should have exactly 4 images of every id

    df['merger'] = (df['name'] == 'major_merger').astype(int)  # label column: 1 if major merger, 0 if not

    df.to_csv('catalogs/final_catalog.csv', index=False)

    # drop minor mergers
    df = df[(df['name'] != 'minor_merger')]

    for img_size in [256]:
        train_loc = 'tfrecord/illustris_no-minor_s{}_rescaled_train.tfrecord'.format(img_size)
        test_loc = 'tfrecord/illustris_no-minor_s{}_rescaled_test.tfrecord'.format(img_size)
        split_carefully(df, 'merger', train_loc, test_loc, img_size, columns_to_save=[])
