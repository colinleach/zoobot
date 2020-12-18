import os
import glob
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from zoobot import label_metadata
from zoobot.active_learning import run_estimator_config
from zoobot.estimators import losses, input_utils

from zoobot.estimators import efficientnet


def prediction_to_row(prediction, png_loc, label_cols):
    row = {
        'png_loc': png_loc
    }
    for n, col in enumerate(label_cols):
        answer = label_cols[n]
        row[answer + '_concentration'] = json.dumps(list(prediction[n].astype(float)))
        # row[answer + '_concentration_mean'] = float(prediction[n].mean())
    return row


# https://stackoverflow.com/questions/62544528/tensorflow-decodejpeg-expected-image-jpeg-png-or-gif-got-unknown-format-st?rq=1
def load_image_file(loc, mode='png'):
    # specify mode explicitly to avoid graph tracing issues
    image = tf.io.read_file(loc)
    if mode == 'png':
        image = tf.image.decode_png(image)
    elif mode == 'jpeg':  # rename jpg to jpeg or validation checks in decode_jpg will fail
        image = tf.image.decode_jpeg(image)
    else:
        raise ValueError(f'Image filetype mode {mode} not recognised')
    return tf.image.convert_image_dtype(image, tf.float32)


def resize_image_batch_with_tf(batch, size):
    return tf.image.resize(batch, (size, size), method=tf.image.ResizeMethod.LANCZOS3, antialias=True)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    local = os.path.isdir('/home/walml')  # only appropriate for me

    # driver errors if you don't include this
    if local:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # parameters (many)

    # decals cols
    questions = label_metadata.decals_questions
    version = 'decals'
    label_cols = label_metadata.decals_label_cols
    
    # initial_size = 64
    initial_size = 300
    crop_size = int(initial_size * 0.75)
    final_size = 224

    channels = 3
    if local:
        batch_size = 8  # largest that fits on laptop  @ 224 pix
        n_samples = 5
    else:
        batch_size = 128  # 16 for B7, 128 for B0
        n_samples = 5

    if local:
        # catalog_loc = 'data/decals/decals_master_catalog.csv'
        # checkpoint_dir = 'results/temp/decals_n2_allq_m0/in_progress'
        checkpoint_dir = 'results/debug/in_progress'
        folder_to_predict = '/media/walml/beta/decals/png_native/dr5/J000'
        save_loc = 'temp/local_debugging.csv'
        file_format = 'png'
    else:  # on ARC HPC system
        data_dir = os.environ['DATA']
        logging.info(data_dir)
        # catalog_loc = f'{data_dir}/repos/zoobot/data/decals/decals_master_catalog_arc.csv'
        model_name = 'decals_dr_train_labelled_m1'
        checkpoint_dir = f'{data_dir}/repos/zoobot/results/{model_name}/in_progress'
        # folder_to_predict = f'{data_dir}/png_native/dr5/J000'
        folder_to_predict = '/data/phys-zooniverse/chri5177/galaxy_zoo/decals/dr1_dr2/png/dr2/standard'
        # folder_to_predict = f'{data_dir}/repos/zoobot/data/decals/temp/J000'
        file_format = 'jpeg'
        folder_name = 'dr2'
        save_loc = f'{data_dir}/repos/zoobot/results/folder_{folder_name}_model_{model_name}_predictions.csv'

    # go
    
    logging.info(f'{checkpoint_dir}, {folder_to_predict}, {save_loc}, {local}, {n_samples}, {batch_size}')
    start = time.time()
    logging.info('Starting at: {}'.format(start))

    schema = losses.Schema(label_cols, questions, version=version)

    # catalog = pd.read_csv(catalog_loc, dtype={'subject_id': str})  # original catalog

    assert os.path.isdir(folder_to_predict)
    # png_paths = list(Path('/media/walml/beta/decals/dr5/png_native').glob('*/**.png'))
    png_paths = list(Path(folder_to_predict).glob('*.{}'.format(file_format)))  # not recursive
    assert png_paths
    logging.info('Images to predict on: {}'.format(len(png_paths)))

    # check they exist
    missing_paths = [path for path in png_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f'Missing {len(missing_paths)} images e.g. {missing_paths[0]}')

    path_ds = tf.data.Dataset.from_tensor_slices([str(path) for path in png_paths])

    png_ds = path_ds.map(lambda x: load_image_file(x, mode=file_format)) 
    png_ds = png_ds.batch(batch_size, drop_remainder=False)

    png_ds = png_ds.map(lambda x: resize_image_batch_with_tf(x , size=initial_size))   # initial size = after resize from 424 but before crop/zoom
    png_ds = png_ds.map(lambda x: tf.reduce_mean(input_tensor=x, axis=3, keepdims=True))  # greyscale
    png_ds = png_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # print('First path: ', str(png_paths[0]))
    # for path, png in zip(png_paths, png_ds):
    #     print(str(path))
    #     time.sleep(0.1)
    #     print(png.numpy()[0, 0, 0])
    # exit()

    model = run_estimator_config.get_model(schema, initial_size, crop_size, final_size)
    load_status = model.load_weights(checkpoint_dir)
    load_status.assert_nontrivial_match()
    load_status.assert_existing_objects_matched()

    logging.info('Beginning predictions')
    # predictions = model.predict(png_ds.take(1))  # np out
    # print(predictions[:, 0] / predictions[:, :2].sum(axis=1))

    predictions = np.stack([model.predict(png_ds) for n in range(n_samples)], axis=-1)
    logging.info('Predictions complete - {}'.format(predictions.shape))

    data = [prediction_to_row(predictions[n], png_paths[n], label_cols=label_cols) for n in range(len(predictions))]
    predictions_df = pd.DataFrame(data)
    logging.info(predictions_df)

    predictions_df.to_csv(save_loc, index=False)
    logging.info(f'Predictions saved to {save_loc}')

    end = time.time()
    logging.info('Time elapsed: {}'.format(end - start))
