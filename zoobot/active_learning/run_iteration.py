
import os
import logging
import itertools
import sqlite3
import shutil
import argparse
import time

import git
import numpy as np

from zoobot.estimators import run_estimator
from zoobot.active_learning import active_learning, iterations, default_estimator_params, acquisition_utils, create_instructions


def run(instructions, train_callable, acquisition_func):
    """Main active learning training loop. 
    
    Learn with train_callable
    Calculate acquisition functions for each subject in the shards
    Load .fits of top subjects and save to a new shard
    Repeat for instructions.iterations
    After each iteration, copy the model history to new directory and start again
    Designed to work with tensorflow estimators
    
    Args:
        train_callable (func): train a tf model. Arg: list of tfrecord locations
        acquisition_func (func): expecting samples of shape [n_subject, n_sample]
    """
    # clear any leftover mocked labels awaiting collection
    # won't do this in production
    from zoobot.active_learning import mock_panoptes
    if os.path.exists(mock_panoptes.SUBJECTS_REQUESTED):
        os.remove(mock_panoptes.SUBJECTS_REQUESTED)

    assert instructions.ready()
    db = sqlite3.connect(instructions.db_loc)
    all_shard_locs = [os.path.join(instructions.shards.shard_dir, os.path.split(loc)[-1]) for loc in active_learning.get_all_shard_locs(db)]
    shards_iterable = itertools.cycle(all_shard_locs)  # cycle through shards

    iteration_n = 0
    initial_estimator_ckpt = instructions.initial_estimator_ckpt  # for first iteration, the first model is the one passed to ActiveConfig
    initial_db_loc = instructions.db_loc
    initial_train_tfrecords = instructions.shards.train_tfrecord_locs()
    eval_tfrecords = instructions.shards.eval_tfrecord_locs()

    learning_rate = 0.001

    iterations_record = []

    while iteration_n < instructions.n_iterations:

        if iteration_n == 0:
            epochs = 125
        else:
            epochs = 50

        prediction_shards = [next(shards_iterable) for n in range(instructions.shards_per_iter)]

        iteration = iterations.Iteration(
            run_dir=instructions.run_dir, 
            name='iteration_{}'.format(iteration_n), 
            prediction_shards=prediction_shards,
            initial_db_loc=initial_db_loc,
            initial_train_tfrecords=initial_train_tfrecords,
            eval_tfrecords=eval_tfrecords,
            train_callable=train_callable,
            acquisition_func=acquisition_func,
            n_samples=instructions.n_samples,
            n_subjects_to_acquire=instructions.subjects_per_iter,
            initial_size=instructions.shards.size,
            learning_rate=learning_rate,
            initial_estimator_ckpt=initial_estimator_ckpt,  # will only warm start with --warm_start, though
            epochs=epochs)

        # train as usual, with saved_model being placed in estimator_dir
        logging.info('Training iteration {}'.format(iteration_n))
        iteration.run()

        # each of these needs to be saved to disk
        iteration_n += 1
        initial_db_loc = iteration.db_loc
        initial_train_tfrecords = iteration.get_train_records()  # includes newly acquired shards
        initial_estimator_ckpt = iteration.estimators_dir
        iterations_record.append(iteration)  # not needed

        # need to be able to end process here

    return iterations_record


def main(instructions_dir, this_iteration_dir, previous_iteration_dir, test=False):

    instructions = create_instructions.load_instructions(instructions_dir)
    train_callable = create_instructions.load_train_callable(instructions_dir).get()
    acquisition_func = create_instructions.load_acquisition_function(instructions_dir).get()

    if args.test:  # override instructions and do a brief run only
        instructions.use_test_mode()

    run(instructions, train_callable, acquisition_func)

    # finally, tidy up by moving the log into the run directory
    # could not be create here because run directory did not exist at start of script
    if os.path.exists(log_loc):  # temporary workaround for disappearing log
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        shutil.move(log_loc, os.path.join(args.run_dir, '{}.log'.format(sha)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Execute single active learning iteration')
    parser.add_argument('--instructions_dir', dest='instructions_dir', type=str,
                    help='Directory with instructions for the execution of an iteration (e.g. shards_per_iter)')
    parser.add_argument('--this_iteration_dir', dest='this_iteration_dir', type=str,
                    help='Directory to save this iteration')
    parser.add_argument('--previous_iteration_dir', dest='previous_iteration_dir_dir', type=str,
                    help='Directory with previously-executed iteration from which to begin (if provided)')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='Only do a minimal iteration to verify that everything works')

    args = parser.parse_args()


    log_loc = 'create_instructions_{}.log'.format(time.time())

    logging.basicConfig(
        filename=log_loc,
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    main(args.instructions_dir, args.this_iteration_dir, args.previous_iteration_dir, args.test)

    # TODO move to simulation controller
    # analysis.show_subjects_by_iteration(iterations_record[-1].get_train_records(), 15, active_config.shards.size, 3, os.path.join(active_config.run_dir, 'subject_history.png'))

