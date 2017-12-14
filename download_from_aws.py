from tqdm import tqdm
import functools
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image
from urllib.request import urlretrieve
import pandas as pd


def download_png_threaded(catalog, png_dir, overwrite=False):

    pbar = tqdm(total=len(catalog), unit=' images created')

    catalog['png_loc'] = [get_png_loc(png_dir, catalog.iloc[index]) for index in range(len(catalog))]

    download_params = {
        'overwrite': overwrite,
        'pbar': pbar
    }
    download_images_partial = functools.partial(download_images, **download_params)

    pool = ThreadPool(30)
    pool.map(download_images_partial, catalog.iterrows())
    pbar.close()
    pool.close()
    pool.join()

    # list(map(download_images_partial, catalog.iterrows()))

    catalog = check_images_are_downloaded(catalog)

    print("\n{} total galaxies".format(len(catalog)))
    print("{} png are downloaded".format(np.sum(catalog['png_ready'])))

    return catalog


def get_png_loc(png_dir, galaxy):
    return '{}/{}.png'.format(png_dir, galaxy['dr7objid'])


def download_images(galaxy, overwrite, max_attempts=5, pbar=None):

    # TODO Temporary fix for iterrows
    galaxy = galaxy[1]

    png_loc = galaxy['png_loc']

    if not png_downloaded_correctly(png_loc) or overwrite:
        attempt = 0
        while attempt < max_attempts:
            try:
                urlretrieve(galaxy['location'], galaxy['png_loc'])
                assert png_downloaded_correctly(png_loc)
                break
            except Exception as err:
                print(err, 'on galaxy {}, attempt {}'.format(galaxy['dr7objid'], attempt))
                attempt += 1

    if pbar:
        pbar.update()


def png_downloaded_correctly(png_loc):
    try:
        im = Image.open(png_loc)
        return True
    except:
        return False


def check_images_are_downloaded(catalog):
    catalog['png_ready'] = np.zeros(len(catalog), dtype=bool)

    for row_index, galaxy in tqdm(catalog.iterrows(), total=len(catalog), unit=' images checked'):
        png_loc = galaxy['png_loc']
        catalog['png_ready'][row_index] = png_downloaded_correctly(png_loc)

    return catalog


if __name__ == '__main__':

    nrows = None
    png_dir = '/Volumes/EXTERNAL/gz2/png'
    overwrite = True

    catalog_dir = '/data/galaxy_zoo/gz2/subjects'
    labels_loc = '{}/all_labels.csv'.format(catalog_dir)

    labels = pd.read_csv(labels_loc, nrows=nrows)

    labels = download_png_threaded(labels, png_dir, overwrite)

    labels.to_csv('{}/all_labels_downloaded.csv'.format(catalog_dir), index=None)
