import pandas as pd

from astropy.io import fits
from PIL import Image

from zoobot.tfrecord import catalog_to_tfrecord


def load_illustris_as_pil(subject):  # should be dict with 'fits_loc' component
    img = fits.getdata(subject['fits_loc'])
    img = img - img.min()
    img = img / img.max()
    return Image.fromarray(img * 255)


def split_carefully(df, label_col, train_loc, test_loc, img_size, columns_to_save, train_test_fraction=0.8):
    # TODO this should be refactored into catalog utils for general case
    # Don't split blindly. Make sure each galaxy id is in only one set.
    df = df.sort_values('id')

    train_test_split = int(train_test_fraction * len(df))
    train_df = df[:train_test_split].copy().sample(frac=1).reset_index(drop=True)
    test_df = df[train_test_split:].copy().sample(frac=1).reset_index(drop=True)

    train_df.to_csv(train_loc + '.csv')  # ugly but effective
    test_df.to_csv(test_loc + '.csv')

    assert not train_df.empty
    assert not test_df.empty

    catalog_to_tfrecord.write_image_df_to_tfrecord(train_df, label_col, train_loc, img_size, columns_to_save, append=False, source='fits', load_fits_as_pil=load_illustris_as_pil)
    catalog_to_tfrecord.write_image_df_to_tfrecord(test_df, label_col, test_loc, img_size, columns_to_save, append=False, source='fits', load_fits_as_pil=load_illustris_as_pil)


if __name__ == '__main__':
    catalog_loc = 'catalogs/illustris_galaxies.csv'
    df = pd.read_csv(catalog_loc)
    df['merger'] = (df['name'] == 'major_merger').astype(int)  # label column: 1 if major merger, 0 if not
    df = df[(df['name'] == 'major_merger') | (df['name'] == 'random_sample')]

    img_size = 64
    train_loc = 'tfrecord/illustris_major_s{}_train.tfrecord'.format(img_size)
    test_loc = 'tfrecord/illustris_major_s{}_test.tfrecord'.format(img_size)
    split_carefully(df, 'merger', train_loc, test_loc, img_size, columns_to_save=[])
