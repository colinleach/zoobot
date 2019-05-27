
import os
import logging
import itertools
import sqlite3
import shutil
import argparse
import time
from collections import namedtuple
import json

import git
import numpy as np

from zoobot.estimators import run_estimator
from zoobot.active_learning import active_learning, iterations, default_estimator_params, acquisition_utils, create_instructions, mock_panoptes

InitialState = namedtuple(
    'InitialState',
    [
        'iteration_dir',
        'iteration_n',
        'initial_estimator_ckpt',
        'initial_train_tfrecords',
        'initial_db_loc',
        'prediction_shards',
        'learning_rate',
        'epochs'
    ]
)

FinalState = namedtuple(
    'FinalState',
    [
        'iteration_n',
        'estimators_dir',
        'train_records',
        'db_loc',
    ]
)

def run(initial_state, instructions, train_callable, acquisition_func, oracle):
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
    iteration = iterations.Iteration(
        iteration_dir=initial_state.iteration_dir, 
        prediction_shards=initial_state.prediction_shards,
        initial_db_loc=initial_state.initial_db_loc,
        initial_train_tfrecords=initial_state.initial_train_tfrecords,
        eval_tfrecords=instructions.shards.eval_tfrecord_locs(),
        train_callable=train_callable,
        acquisition_func=acquisition_func,
        n_samples=instructions.n_samples,
        n_subjects_to_acquire=instructions.subjects_per_iter,
        initial_size=instructions.shards.size,
        learning_rate=initial_state.learning_rate,
        initial_estimator_ckpt=initial_state.initial_estimator_ckpt,  # will only warm start with --warm_start, though
        epochs=initial_state.epochs,
        oracle=oracle
        )

    # train as usual, with saved_model being placed in estimator_dir
    logging.info('Training iteration {}'.format(initial_state.iteration_n))
    iteration.run()

    final_state = FinalState(
        iteration_n=initial_state.iteration_n,
        estimators_dir=iteration.estimators_dir,  # to become initial_estimator_ckpt'
        train_records=iteration.get_train_records(),  # initial_train_tfrecords
        db_loc=iteration.db_loc
    )
    return final_state


def get_initial_state(instructions, this_iteration_dir, previous_iteration_dir):
    """Decide, based on previous final state, what to tweak for the next iteration
    
    Args:
        instructions ([type]): [description]
        this_iteration_dir ([type]): [description]
        previous_iteration_dir ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    if (previous_iteration_dir is None) or (previous_iteration_dir is ""):
        this_iteration_n = 0
        initial_state = InitialState(
            iteration_dir=this_iteration_dir,  # duplication
            iteration_n=0,  # duplication
            initial_estimator_ckpt=instructions.initial_estimator_ckpt,  # for first iteration, the first model is the one passed to ActiveConfig
            initial_db_loc=instructions.db_loc,
            initial_train_tfrecords=instructions.shards.train_tfrecord_locs(),
            prediction_shards=get_prediction_shards(this_iteration_n, instructions),  # duplication
            learning_rate=get_learning_rate(this_iteration_n),  # duplication
            epochs=get_epochs(this_iteration_n)  # duplication
        )
    else:
        with open(os.path.join(previous_iteration_dir, 'final_state.json'), 'r') as f:  # coupled to saving of final state
            previous_final_state = FinalState(**json.load(f))
            this_iteration_n = previous_final_state.iteration_n + 1
            initial_state = InitialState(
                iteration_dir=this_iteration_dir,  # duplication
                iteration_n=this_iteration_n,  # duplication
                initial_train_tfrecords=previous_final_state.train_records,
                initial_estimator_ckpt=previous_final_state.estimators_dir,
                initial_db_loc=previous_final_state.db_loc,
                prediction_shards=get_prediction_shards(this_iteration_n, instructions),  # duplication
                learning_rate=get_learning_rate(this_iteration_n),  # duplication
                epochs=get_epochs(this_iteration_n)  # duplication
            )
    return initial_state


def save_final_state(final_state, save_dir):
    with open(os.path.join(save_dir, 'final_state.json'), 'w') as f:
        json.dump(final_state._asdict(), f)


def get_prediction_shards(iteration_n, instructions):
    db = sqlite3.connect(instructions.db_loc)
    all_shard_locs = [os.path.join(instructions.shards.shard_dir, os.path.split(loc)[-1]) for loc in active_learning.get_all_shard_locs(db)]
    shards_iterable = itertools.cycle(all_shard_locs)  # cycle through shards
    for n in range(iteration_n + 1):  # get next shards once for iteration_n = 0, etc.
        prediction_shards = [next(shards_iterable) for n in range(instructions.shards_per_iter)]
    return prediction_shards


def get_learning_rate(iteration_n):
    return 0.001  # may be reduced to 0.0001 from 0.001 (for first bar model, but not for smooth)


def get_epochs(iteration_n):
    if iteration_n == 0:
        return 1000  # let's see how we do with 1 iteration only
    else:
        return 50


def main(instructions_dir, this_iteration_dir, previous_iteration_dir, test=False):

    instructions = create_instructions.load_instructions(instructions_dir)
    assert instructions.ready()
    train_callable = create_instructions.load_train_callable(instructions_dir).get()
    acquisition_func = create_instructions.load_acquisition_func(instructions_dir).get()

    if test:  # override instructions and do a brief run only
        instructions.use_test_mode()
        train_callable.test = True

    oracle = mock_panoptes.load_oracle(instructions_dir)  # decoupled whether real or simulated
    initial_state = get_initial_state(instructions, this_iteration_dir, previous_iteration_dir)
    final_state = run(initial_state, instructions, train_callable, acquisition_func, oracle)
    save_final_state(final_state, this_iteration_dir)

    # finally, tidy up by moving the log into the run directory
    # could not be create here because run directory did not exist at start of script
    if os.path.exists(log_loc):  # temporary workaround for disappearing log
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        shutil.move(log_loc, os.path.join(args.this_iteration_dir, '{}.log'.format(sha)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Execute single active learning iteration')
    parser.add_argument('--instructions-dir', dest='instructions_dir', type=str,
                    help='Directory with instructions for the execution of an iteration (e.g. shards_per_iter)')
    parser.add_argument('--this-iteration-dir', dest='this_iteration_dir', type=str,
                    help='Directory to save this iteration')
    parser.add_argument('--previous-iteration-dir', dest='previous_iteration_dir', type=str,
                    help='Directory with previously-executed iteration from which to begin (if provided)')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='Only do a minimal iteration to verify that everything works')
    args = parser.parse_args()

    log_loc = 'run_iteration_{}.log'.format(time.time())

    logging.basicConfig(
        filename=log_loc,
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    main(args.instructions_dir, args.this_iteration_dir, args.previous_iteration_dir, args.test)

    # TODO move to simulation controller
    # analysis.show_subjects_by_iteration(iterations_record[-1].get_train_records(), 15, active_config.shards.size, 3, os.path.join(active_config.run_dir, 'subject_history.png'))

