import os
import glob
import json
import logging

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

    logging.basicConfig(level=logging.INFO)

    local = os.path.isdir('/home/walml')

    # driver errors if you don't include this
    if local:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # parameters (many)

    # decals cols
    questions = label_metadata.decals_questions
    version = 'decals'
    label_cols = label_metadata.decals_label_cols

    # gz2 cols
    # questions = label_metadata.gz2_partial_questions
    # version = 'gz2'
    # label_cols = label_metadata.gz2_partial_label_cols
    
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
        catalog_loc = 'data/gz2/gz2_master_catalog.csv'
        # tfrecord_locs = glob.glob(f'/home/walml/repos/zoobot/results/temp/decals_n2_allq_m0_eval_shards/*.tfrecord')
        tfrecord_locs = glob.glob(f'/home/walml/repos/zoobot/results/temp/gz2_all_actual_sim_2p5_unfiltered_300_eval_shards/*.tfrecord')
        # previously: add log to get best model
        checkpoint_dir = 'results/temp/all_actual_sim_2p5_unfiltered_300_small_first_baseline_1q_effnetv2/models/final'
        save_loc = 'temp/all_actual_sim_2p5_unfiltered_300_small_first_baseline_1q_effnetv2.csv'
    else:
        data_dir = os.environ['DATA']
        logging.info(data_dir)
        catalog_loc = f'{data_dir}/repos/zoobot/data/decals/decals_master_catalog.csv'
        shard_name = 'decals_dr_full'
        model_name = 'decals_dr_full_reparam'  # and decals_dr_full_reparam
        output_name = model_name + '_eval_predictions'
        # tfrecord_locs = glob.glob(f'{data_dir}/repos/zoobot/data/decals/shards/all_2p5_unfiltered_n2/eval_shards/*.tfrecord')

        # subdirs_to_search = ['', 'train_shards', 'eval_shards']
        subdirs_to_search = ['eval_shards']  # eval only
        dirs_to_search = [os.path.join(f'{data_dir}/repos/zoobot/data/decals/shards/{shard_name}', subdir) for subdir in subdirs_to_search]
        tfrecord_locs = []
        for d in dirs_to_search:
            tfrecord_locs = tfrecord_locs + glob.glob(os.path.join(d, '*.tfrecord'))  # concat lists

        checkpoint_dir = f'{data_dir}/repos/zoobot/results/{model_name}/in_progress'
        save_loc = f'{data_dir}/repos/zoobot/results/{output_name}.csv'

    # go

    logging.info(f'{catalog_loc}, {tfrecord_locs}, {checkpoint_dir}, {save_loc}, {local}, {n_samples}, {batch_size}')
    logging.info(len(tfrecord_locs))

    schema = losses.Schema(label_cols, questions, version=version)

    catalog = pd.read_csv(catalog_loc, dtype={'subject_id': str})  # original catalog

    eval_config = run_estimator_config.get_eval_config(
        tfrecord_locs, [], batch_size, initial_size, final_size, channels  # label cols = [] as we don't know them, in general
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

    logging.info('Beginning predictions')
    predictions = np.stack([model.predict(dataset) for n in range(n_samples)], axis=-1)
    logging.info('Predictions complete - {}'.format(predictions.shape))

    data = [prediction_to_row(predictions[n], id_strs[n]) for n in range(len(predictions))]
    predictions_df = pd.DataFrame(data)

    # if version == 'decals':
    #     catalog['iauname'] = catalog['iauname'].astype(str)  # or dr7objid

    df = pd.merge(catalog, predictions_df, how='inner', on='id_str')
    assert len(df) == len(predictions_df)

    df.to_csv(save_loc, index=False)
    logging.info(f'Predictions saved to {save_loc}')
