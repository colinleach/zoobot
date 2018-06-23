import logging

import tensorflow as tf
import pandas as pd

from zoobot.estimators import estimator_funcs, bayesian_estimator_funcs, run_estimator, input_utils
from zoobot import panoptes_to_tfrecord


def get_stratify_probs_from_csv(input_config):
    subject_df = pd.read_csv(train_config.tfrecord_loc + '.csv')
    return [1. - subject_df[input_config.label_col].mean(), subject_df[input_config.label_col].mean()]


image_dim = 424
label_split_value = '0.4'
run_name = 'bayesian_panoptes_featured_s{}_l{}_saver'.format(image_dim, label_split_value)

logging.basicConfig(
    filename=run_name + '.log',
    format='%(asctime)s %(message)s',
    filemode='w',
    level=logging.INFO)

new_tfrecords = False

if new_tfrecords:
    panoptes_to_tfrecord.save_panoptes_to_tfrecord()


run_config = run_estimator.RunEstimatorConfig(
    image_dim=image_dim,
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
    tfrecord_loc='data/panoptes_featured_s{}_l{}_train.tfrecord'.format(image_dim, label_split_value),
    label_col=run_config.label_col,
    stratify=True,
    stratify_probs=None,
    transform=True,
    rotation_range=90,
    height_shift_range=5,
    width_shift_range=5,
    zoom_range=[0.95, 1.05],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='wrap',
    cval=0.,
    batch_size=run_config.batch_size,
    image_dim=run_config.image_dim,
    channels=run_config.channels,
)
train_config.stratify_probs = get_stratify_probs_from_csv(train_config)


eval_config = input_utils.InputConfig(
    name='eval',
    tfrecord_loc='data/panoptes_featured_s{}_l{}_test.tfrecord'.format(image_dim, label_split_value),
    label_col=run_config.label_col,
    stratify=True,
    stratify_probs=None,
    transform=True,
    rotation_range=90,
    height_shift_range=5,
    width_shift_range=5,
    zoom_range=[0.95, 1.05],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='wrap',
    cval=0.,
    batch_size=run_config.batch_size,
    image_dim=run_config.image_dim,
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
    image_dim=run_config.image_dim
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
