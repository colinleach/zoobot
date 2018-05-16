import logging

from zoobot.estimators import estimator_funcs, bayesian_estimator_funcs, estimator_params, run_estimator
from zoobot import panoptes_to_tfrecord

image_dim = 28
run_name = 'chollet_panoptes_featured_bayesian_{}'.format(image_dim)

logging.basicConfig(
    filename=run_name + '.log',
    format='%(asctime)s %(message)s',
    filemode='w',
    level=logging.INFO)

new_tfrecords = False

if new_tfrecords:
    panoptes_to_tfrecord.save_panoptes_to_tfrecord()

params = estimator_params.default_params()
# TODO optionally refactor these out of panoptes_to_tfrecord?

params['train_loc'] = 'data/panoptes_calibration_featured_{}_train.tfrecord'.format(image_dim)
params['test_loc'] = 'data/panoptes_calibration_featured_{}_test.tfrecord'.format(image_dim)

params['label_col'] = 'smooth-or-featured_prediction-encoded'

params['epochs'] = 1000
params['train_stratify'] = False
params['test_stratify'] = False
params['image_dim'] = image_dim
params['log_dir'] = 'runs/{}'.format(run_name)
params['log_freq'] = 10
params['logging_hooks'] = bayesian_estimator_funcs.logging_hooks(params)  # TODO coupled code, refactor

params.update(estimator_params.default_four_layer_architecture())

logging.info('Parameters used: ')
for key, value in params.items():
    logging.info('{}: {}'.format(key, value))

# model_fn = estimator_funcs.four_layer_binary_classifier
model_fn = bayesian_estimator_funcs.four_layer_binary_classifier

run_estimator.run_estimator(model_fn, params)
