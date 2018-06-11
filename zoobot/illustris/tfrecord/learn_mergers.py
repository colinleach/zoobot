import logging

from zoobot.estimators import bayesian_estimator_funcs, estimator_params, run_estimator

image_dim = 64
run_name = 'illustris_s{}'.format(image_dim)

logging.basicConfig(
    filename=run_name + '.log',
    format='%(asctime)s %(message)s',
    filemode='w',
    level=logging.INFO)

params = estimator_params.default_params()
# TODO optionally refactor these out of panoptes_to_tfrecord?

params['train_loc'] = 'zoobot/illustris/tfrecord/illustris_major_s{}_train.tfrecord'.format(image_dim)
params['test_loc'] = 'zoobot/illustris/tfrecord/illustris_major_s{}_test.tfrecord'.format(image_dim)

params['label_col'] = 'merger'

params['epochs'] = 5000
params['train_batches'] = 30
params['train_stratify'] = True
params['test_stratify'] = True
params['image_dim'] = image_dim
params['log_dir'] = 'runs/{}'.format(run_name)
params['log_freq'] = 10
params['logging_hooks'] = bayesian_estimator_funcs.logging_hooks(params)  # TODO coupled code, refactor
params['save_freq'] = 100

params['channels'] = 1

params.update(estimator_params.default_four_layer_architecture())

logging.info('Parameters used: ')
for key, value in params.items():
    logging.info('{}: {}'.format(key, value))

model_fn = bayesian_estimator_funcs.four_layer_binary_classifier

run_estimator.run_estimator(model_fn, params)
