import logging

import tensorflow as tf
import pandas as pd

from zoobot.estimators import bayesian_estimator_funcs, run_estimator, input_utils
from zoobot import panoptes_to_tfrecord


def get_stratify_probs_from_csv(input_config: input_utils.InputConfig) -> list:
    subject_df = pd.read_csv(train_config.tfrecord_loc + '.csv')
    return [1. - subject_df[input_config.label_col].mean(), subject_df[input_config.label_col].mean()]


initial_size = 128
final_size = 64
label_split_value = '0.4'
run_name = 'bayesian_panoptes_featured_s{}_l{}_saver'.format(initial_size, label_split_value)

logging.basicConfig(
    filename=run_name + '.log',
    format='%(asctime)s %(message)s',
    filemode='w',
    level=logging.INFO)

new_tfrecords = False

if new_tfrecords:
    panoptes_to_tfrecord.save_panoptes_to_tfrecord()


run_config = run_estimator.RunEstimatorConfig(
    initial_size=initial_size,
    final_size=final_size,
    channels=3,
    label_col='smooth-or-featured_prediction-encoded',
    epochs=50,
    train_batches=30,
    eval_batches=3,
    batch_size=128,
    min_epochs=0,
    early_stopping_window=10,
    max_sadness=4.,
    log_dir='runs/{}'.format(run_name),
    save_freq=10
)


train_config = input_utils.InputConfig(
    name='train',
    tfrecord_loc='data/panoptes_featured_s{}_l{}_train.tfrecord'.format(initial_size, label_split_value),
    label_col=run_config.label_col,
    stratify=True,
    stratify_probs=None,
    geometric_augmentation=True,
    max_zoom=[0.95, 1.05],
    fill_mode='wrap',
    batch_size=run_config.batch_size,
    initial_size=run_config.initial_size,
    final_size=run_config.final_size,
    channels=run_config.channels,
)
train_config.stratify_probs = get_stratify_probs_from_csv(train_config)


eval_config = input_utils.InputConfig(
    name='eval',
    tfrecord_loc='data/panoptes_featured_s{}_l{}_test.tfrecord'.format(initial_size, label_split_value),
    label_col=run_config.label_col,
    stratify=True,
    stratify_probs=None,
    geometric_augmentation=True,
    max_zoom=[0.95, 1.05],
    fill_mode='wrap',
    batch_size=run_config.batch_size,
    initial_size=run_config.initial_size,
    final_size=run_config.final_size,
    channels=run_config.channels,
)
eval_config.stratify_probs = get_stratify_probs_from_csv(eval_config)


model = bayesian_estimator_funcs.BayesianBinaryModel(
    learning_rate=0.001,
    optimizer=tf.train.AdamOptimizer,
    conv1_filters=32,
    conv1_kernel=1,
    conv2_filters=32,
    conv2_kernel=3,
    conv3_filters=16,
    conv3_kernel=3,
    dense1_units=128,
    dense1_dropout=0.5,
    log_freq=10,
    image_dim=run_config.initial_size
)

run_config.train_config = train_config
run_config.eval_config = eval_config
run_config.model = model
assert run_config.is_ready_to_train()

# TODO update logging using object calls
# logging.info('Parameters used: ')
# for key, value in params.items():
#     logging.info('{}: {}'.format(key, value))


run_estimator.run_estimator(run_config)
