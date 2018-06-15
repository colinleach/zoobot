import logging
import time

import numpy as np
from matplotlib import pyplot as plt

from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
from skopt.space import Integer

from zoobot.estimators import bayesian_estimator_funcs, estimator_params, run_estimator


def train_on_illustris(custom_params):

    image_dim = 256
    runtime = int(time.time())
    run_name = 'illustris_s{}_adam_{}'.format(image_dim, runtime)

    params = estimator_params.default_params()
    params.update(estimator_params.default_four_layer_architecture())

    params['train_loc'] = 'zoobot/illustris/tfrecord/illustris_no-minor_s{}_rescaled_train.tfrecord'.format(image_dim)
    params['test_loc'] = 'zoobot/illustris/tfrecord/illustris_no-minor_s{}_rescaled_test.tfrecord'.format(image_dim)
    params['channels'] = 1
    params['label_col'] = 'merger'

    params['epochs'] = 5000  # total loops
    params['train_batches'] = 1000  # training steps to take per loop WRONG will complete dataset or stop early?
    params['train_stratify'] = True
    params['test_stratify'] = True
    params['image_dim'] = image_dim
    params['log_dir'] = 'runs/{}'.format(run_name)
    params['log_freq'] = 100
    params['logging_hooks'] = bayesian_estimator_funcs.logging_hooks(params)  # TODO coupled code, refactor
    params['save_freq'] = 100

    params['min_epochs'] = 20
    params['early_stopping_window'] = 10
    params['max_sadness'] = 4.

    params['conv1_filters'] = 32
    params['conv1_kernel'] = 1

    # params['pool1_size'] = 2
    # params['pool1_strides'] = 2

    params['conv2_filters'] = 32
    params['conv2_kernel'] = 3

    # params['pool2_size'] = 2
    # params['pool2_strides'] = 2

    params['conv3_filters'] = 16
    params['conv3_kernel'] = 3

    # params['pool3_size'] = 2
    # params['pool3_strides'] = 2

    params['dense1_units'] = 128
    params['dense1_dropout'] = 0.5

    params.update(custom_params)

    logging.info('Parameters used: ')
    for key, value in params.items():
        logging.info('{}: {}'.format(key, value))

    model_fn = bayesian_estimator_funcs.four_layer_binary_classifier

    return run_estimator.run_estimator(model_fn, params)


def grid_search_on_illustris():

    # TODO: use x0, y0 to provide history

    # see https://scikit-optimize.github.io/notebooks/hyperparameter-optimization.html
    # param_limits = [Integer(128, 1064, name='dense1_units')]

    param_limits = [
        Integer(128, 1064, name='dense1_units'),
        Integer(8, 32, name='conv1_filters'),
        Integer(1, 3, name='conv1_kernel'),
        Integer(8, 32, name='conv2_filters'),
        Integer(1, 3, name='conv2_kernel'),
        Integer(16, 64, name='conv3_filters'),
        Integer(1, 3, name='conv3_kernel'),
    ]

    @use_named_args(param_limits)
    def grid_search_objective_function(**params):
        logging.info('Trying out custom params: {}'.format(params))
        loss_history = train_on_illustris(custom_params=params)
        return np.mean(loss_history[-5:])

    # only supports continuous variables
    res = gp_minimize(grid_search_objective_function,  # the function to minimize
                      param_limits,  # the bounds on dense1_units
                      acq_func="gp_hedge",  # the acquisition function
                      n_calls=300,  # the number of evaluations of f
                      n_random_starts=5)  # random initialization points
    logging.info(res)
    logging.info('Best hyperparams: {}'.format(res.fun))
    fig, ax = plt.subplots(nrows=1)
    plot_convergence(res, ax=ax)
    fig.savefig('zoobot/illustris/convergence_{}.png'.format(time.time()))


if __name__ == '__main__':

    logging.basicConfig(
        filename='zoobot/illustris/grid_search_{}.log'.format(int(time.time())),
        format='%(asctime)s %(message)s',
        filemode='w',
        level=logging.INFO)

    train_on_illustris({})
    # grid_search_on_illustris()

