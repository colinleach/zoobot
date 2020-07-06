import os
import glob
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from zoobot import label_metadata
from zoobot.active_learning import run_estimator_config
from zoobot.estimators import losses, input_utils


def prediction_to_row(prediction, id_str):
    row = {
        'id_str': id_str
    }
    for n, col in enumerate(label_cols):
        answer = label_cols[n]
        row[answer + '_concentration'] = json.dumps(list(prediction[n].astype(float)))
        row[answer + '_concentration_mean'] = float(prediction[n].mean())
    return row


if __name__ == '__main__':

    local = os.path.isdir('/home/walml')

    # driver errors if you don't include this
    if local:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # parameters (many)

    # catalog_loc = 'data/latest_labelled_catalog.csv
    # catalog_loc = 'data/decals/decals_master_catalog.csv'
    # catalog_loc = 'data/gz2/gz2_master_catalog.csv'


    # decals cols
    questions = label_metadata.decals_questions
    version = 'decals'
    label_cols = label_metadata.decals_label_cols

    # gz2 cols
    # questions = label_metadata.gz2_questions
    # version = 'gz2'
    # label_cols = label_metadata.gz2_label_cols
    
    initial_size = 300
    crop_size = int(initial_size * 0.75)
    final_size = 224
    channels = 3
    if local:
        batch_size = 8  # or 128
        n_samples = 2
    else:
        batch_size = 128  # or 128
        n_samples = 2

    if local:
        catalog_loc = 'data/decals/decals_master_catalog_arc.csv'
        tfrecord_locs = glob.glob(f'/home/walml/repos/zoobot/results/temp/decals_n2_allq_m0_eval_shards/*.tfrecord')
        checkpoint_dir = 'results/temp/decals_n2_allq_m0/models/final'
        save_loc = 'temp/temp.csv'
    else:
        data_dir = os.environ['DATA']
        catalog_loc = f'{data_dir}/repos/zoobot/data/decals/all_2p5_unfiltered_n2_arc.csv'
        tfrecord_locs = glob.glob(f'{data_dir}/repos/zoobot/decals/shards/all_2p5_unfiltered_n2/eval_shards/*.tfrecord')
        checkpoint_dir = f'{data_dir}/repos/zoobot/data/experiments/live/decals_n2_allq_m0/models/final'
        save_loc = f'{data_dir}/repos/zoobot/results/decals_n2_allq_m0_eval.csv'



    # go

    schema = losses.Schema(label_cols, questions, version=version)

    catalog = pd.read_csv(catalog_loc, dtype={'subject_id': str})  # original catalog

    eval_config = run_estimator_config.get_eval_config(
        tfrecord_locs, label_cols, batch_size, initial_size, final_size, channels
    )
    eval_config.drop_remainder = False
    dataset = input_utils.get_input(config=eval_config)

    feature_spec = input_utils.get_feature_spec({'id_str': 'string'})
    id_str_dataset = input_utils.get_dataset(
        tfrecord_locs, feature_spec, batch_size=1, shuffle=False, repeat=False, drop_remainder=False
    )
    id_strs = [str(d['id_str'].numpy().squeeze())[2:-1] for d in id_str_dataset]

    model = run_estimator_config.get_model(schema, initial_size, crop_size, final_size)
    load_status = model.load_weights(checkpoint_dir)
    load_status.assert_nontrivial_match()
    load_status.assert_existing_objects_matched()

    predictions = np.stack([model.predict(dataset) for n in range(n_samples)], axis=-1)

    data = [prediction_to_row(predictions[n], id_strs[n]) for n in range(len(predictions))]
    predictions_df = pd.DataFrame(data)

    catalog['iauname'] = catalog['iauname'].astype(str)  # or dr7objid

    df = pd.merge(catalog, predictions_df, how='inner', on='id_str')
    assert len(df) == len(predictions_df)

    df.to_csv(save_loc, index=False)
