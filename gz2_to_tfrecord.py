import os

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from create_tfrecord import image_to_tfrecord


size = 64
columns_to_save = [
    # 't04_spiral_a08_spiral_count',
    #                't04_spiral_a09_no_spiral_count',
                   't04_spiral_a08_spiral_weighted_fraction',
    #                'id',
                   'ra',
                   'dec']

df = pd.read_csv('/data/galaxy_zoo/gz2/subjects/all_labels_downloaded.csv',
                 usecols=columns_to_save + ['png_loc', 'png_ready'],
                 nrows=100)
tfrecord_loc = '/data/galaxy_zoo/gz2/tfrecord/spiral_{}.tfrecord'.format(size)
dimensions = size, size

if os.path.exists(tfrecord_loc):
    os.remove(tfrecord_loc)

for subject_n, subject in tqdm(df.iterrows(), total=len(df), unit=' subjects saved'):
    if subject['png_ready']:
        img = Image.open(subject['png_loc'])
        img.thumbnail(dimensions)  # inplace on img
        matrix = np.array(img)
        # label = int(subject['t04_spiral_a08_spiral_weighted_fraction'])
        label = np.random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        extra_data = {}
        for col in columns_to_save:
            extra_data.update({col: subject[col]})

        # print(matrix)
        # print(label)
        # print(extra_data)
        image_to_tfrecord(matrix, label, tfrecord_loc, extra_data)
