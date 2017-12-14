from tqdm import tqdm
import functools
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image


def download_png_threaded(catalog, png_dir, overwrite=False):

    print(catalog.iloc[0])

    pbar = tqdm(total=len(catalog), unit=' images created')

    catalog['png_loc'] = [get_png_loc(png_dir, catalog[index]) for index in range(len(catalog))]

    download_params = {
        'overwrite': overwrite,
        'pbar': pbar
    }
    download_images_partial = functools.partial(download_images, **download_params)

    pool = ThreadPool(30)
    pool.map(download_images_partial, catalog)
    pbar.close()
    pool.close()
    pool.join()

    catalog = check_images_are_downloaded(catalog)

    print("\n{} total galaxies".format(len(catalog)))
    print("{} fits are downloaded".format(np.sum(catalog['fits_ready'])))
    print("{} jpeg generated".format(np.sum(catalog['jpeg_ready'])))
    print("{} fits have many bad pixels".format(len(catalog) - np.sum(catalog['fits_filled'])))

    return catalog


def get_png_loc(galaxy, png_dir):
    return '{}/{}'.format(png_dir, galaxy['dr7id'])


def download_images(galaxy, overwrite, max_attempts=5, pbar=None):

    png_loc = galaxy['png_loc']

    if not png_downloaded_correctly(png_loc) or overwrite:
        attempt = 0
        downloaded = False
        while attempt < max_attempts:
            try:
                download_png(galaxy['url'], galaxy['png_loc'])
                assert png_downloaded_correctly(png_loc)
                break
            except Exception as err:
                print(err, 'on galaxy {}, attempt {}'.format(galaxy['dr7id'], attempt))
                attempt += 1

    if pbar:
        pbar.update()


def download_png(url, png_loc):
    

def png_downloaded_correctly(png_loc):
    try:
        im = Image.open(png_loc)
        return True
    except:
        return False


def check_images_are_downloaded(catalog):
    catalog['png_ready'] = np.zeros(len(catalog), dtype=bool)

    for row_index, galaxy in tqdm(enumerate(catalog), total=len(catalog), unit=' images checked'):
        png_loc = galaxy['png_loc']
        catalog['png_ready'][row_index] = png_downloaded_correctly(png_loc)

    return catalog

