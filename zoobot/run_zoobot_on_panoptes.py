import logging

import pandas as pd

from zoobot.estimators import estimator_funcs, bayesian_estimator_funcs, estimator_params, run_estimator
from zoobot import panoptes_to_tfrecord

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

params = estimator_params.default_params()
params.update(estimator_params.default_four_layer_architecture())
# TODO optionally refactor these out of panoptes_to_tfrecord?

params['image_dim'] = image_dim
params['train_loc'] = 'data/panoptes_featured_s{}_l{}_train.tfrecord'.format(image_dim, label_split_value)
params['test_loc'] = 'data/panoptes_featured_s{}_l{}_test.tfrecord'.format(image_dim, label_split_value)
params['channels'] = 3
params['label_col'] = 'smooth-or-featured_prediction-encoded'

params['epochs'] = 50
params['train_batches'] = 30
params['eval_batches'] = 3


train_df = pd.read_csv(params['train_loc'] + '.csv')
stratify_prior_probs = [1. - train_df['label'].mean(), train_df['label'].mean()]  # could do in input_utils?
params['stratify_probs'] = stratify_prior_probs  # always specified for rigor, even if stratify is False

# TODO use objects for params and train/eval/predict
params['train'] = {
    'name': 'train',
    'stratify': True,
    'transform': True,
    'rotation_range': 90,
    'height_shift_range': 5,
    'width_shift_range': 5,
    'zoom_range': [0.95, 1.05],
    'horizontal_flip': True,
    'vertical_flip': True,
    'fill_mode': 'wrap',
    'cval': 0.,
    'batch_size': params['batch_size'],
    'image_dim': params['image_dim'],
    'channels': params['channels'],
    'stratify_probs': params['stratify_probs']
}
params['eval'] = {
    'name': 'eval',
    'stratify': True,
    'transform': False,
    'batch_size': params['batch_size'],
    'image_dim': params['image_dim'],
    'channels': params['channels'],
    'stratify_probs': params['stratify_probs']
}

params['min_epochs'] = 0
params['early_stopping_window'] = 10
params['max_sadness'] = 4.


params['log_dir'] = 'runs/{}'.format(run_name)
params['log_freq'] = 10
params['logging_hooks'] = bayesian_estimator_funcs.logging_hooks(params)  # TODO coupled code, refactor
params['save_freq'] = 10

params['conv1_filters'] = 32
params['conv1_kernel'] = 1

params['conv2_filters'] = 32
params['conv2_kernel'] = 3

params['conv3_filters'] = 16
params['conv3_kernel'] = 3

params['dense1_units'] = 128
params['dense1_dropout'] = 0.5

logging.info('Parameters used: ')
for key, value in params.items():
    logging.info('{}: {}'.format(key, value))

# model_fn = estimator_funcs.four_layer_binary_classifier
model_fn = bayesian_estimator_funcs.four_layer_binary_classifier

run_estimator.run_estimator(model_fn, params)
