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
import tensorflow as tf

from zoobot.active_learning import database, iterations, acquisition_utils, create_instructions, oracles, run_estimator_config
from zoobot import label_metadata

InitialState = namedtuple(
    'InitialState',
    [
        'iteration_dir',
        'iteration_n',
        'checkpoints',
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
        'prediction_checkpoints',
        'train_records',
        'db_loc',
    ]
)

def run(initial_state, instructions, fixed_estimator_params, acquisition_func, oracle, questions, label_cols, test=False):
    """Main active learning training loop. 
    
    Calculate acquisition functions for each subject in the shards
    Load .fits of top subjects and save to a new shard
    Repeat for instructions.iterations
    After each iteration, copy the model history to new directory and start again
    Designed to work with tensorflow estimators
    
    Args:
    """

    # override a few parameters if test 
    epochs = initial_state.epochs
    initial_train_tfrecords = initial_state.initial_train_tfrecords
    prediction_shards = initial_state.prediction_shards
    eval_tfrecords = instructions.shards.eval_tfrecord_locs()
    n_samples = instructions.n_samples
    if test:
        epochs = 1
        prediction_shards = prediction_shards[:2]
        n_samples = 2
        eval_tfrecords = eval_tfrecords[:2]

    iteration = iterations.Iteration(
        iteration_dir=initial_state.iteration_dir, 
        prediction_shards=prediction_shards,
        initial_db_loc=initial_state.initial_db_loc,
        initial_train_tfrecords=initial_train_tfrecords,
        eval_tfrecords=eval_tfrecords,
        fixed_estimator_params=fixed_estimator_params,
        acquisition_func=acquisition_func,
        n_samples=n_samples,
        n_subjects_to_acquire=instructions.subjects_per_iter,
        initial_size=instructions.shards.size,
        learning_rate=initial_state.learning_rate,
        checkpoints=initial_state.checkpoints,
        epochs=epochs,
        oracle=oracle,
        questions=questions,
        label_cols=label_cols
    )

    # train as usual, with saved_model being placed in estimator_dir
    logging.info('Training iteration {}'.format(initial_state.iteration_n))
    iteration.run()

    final_state = FinalState(
        iteration_n=initial_state.iteration_n,
        train_records=iteration.get_train_records(),  # initial_train_tfrecords
        prediction_checkpoints=iteration.prediction_checkpoints,
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
    skippable_previous_iteration_dirs = [None, '', ' ', 'None', 'none']
    if previous_iteration_dir in skippable_previous_iteration_dirs:
        this_iteration_n = 0
        logging.info('No previous iteration directory - starting from scratch')
        initial_state = InitialState(
            iteration_dir=this_iteration_dir,  # duplication
            iteration_n=0,  # duplication
            initial_db_loc=instructions.db_loc,  # will copy the db from instructions
            initial_train_tfrecords=instructions.shards.train_tfrecord_locs(),  # only the initial train shards
            prediction_shards=get_prediction_shards(this_iteration_n, instructions),  # duplication
            learning_rate=get_learning_rate(this_iteration_n),  # duplication
            epochs=get_epochs(this_iteration_n),  # duplication
            checkpoints=get_initial_checkpoints(instructions)  # start from [None, None], or pretrained checkpoints if they exist
        )
    else:
        previous_final_state = load_final_state(previous_iteration_dir)
        this_iteration_n = previous_final_state.iteration_n + 1
        initial_state = InitialState(
            iteration_dir=this_iteration_dir,  # duplication
            iteration_n=this_iteration_n,  # duplication
            initial_train_tfrecords=previous_final_state.train_records,  # everything trained on by last iteration, i.e. initial train shards + newly acquired shards (from ALL prev iterations)
            initial_db_loc=previous_final_state.db_loc,  # will copy the db from the last iteration
            prediction_shards=get_prediction_shards(this_iteration_n, instructions),  # duplication
            learning_rate=get_learning_rate(this_iteration_n),  # duplication
            epochs=get_epochs(this_iteration_n),  # duplication
            checkpoints=previous_final_state.prediction_checkpoints  # load the weights used to make the previous predictions
        )
    return initial_state


def load_final_state(iteration_dir):
    logging.info(f'Loading final state from {iteration_dir}')
    with open(os.path.join(iteration_dir, 'final_state.json'), 'r') as f:
        return FinalState(**json.load(f))


def save_final_state(final_state, save_dir):
    with open(os.path.join(save_dir, 'final_state.json'), 'w') as f:
        json.dump(final_state._asdict(), f)


def get_prediction_shards(iteration_n, instructions, timeout=15.0):
    db = sqlite3.connect(instructions.db_loc, timeout=timeout, isolation_level='IMMEDIATE')
    all_shard_locs = [os.path.join(instructions.shards.shard_dir, os.path.split(loc)[-1]) for loc in database.get_all_shard_locs(db)]
    shards_iterable = itertools.cycle(all_shard_locs)  # cycle through shards
    for _ in range(iteration_n + 1):  # get next shards once for iteration_n = 0, etc.
        prediction_shards = [next(shards_iterable) for n in range(instructions.shards_per_iter)]
    return prediction_shards


# TODO may need to set this lower for n>1, when restoring models
def get_learning_rate(iteration_n):
    if iteration_n == 0:
        return 0.001
    else:
        return 0.0002  # for small initial adjustments


def get_epochs(iteration_n):
    if iteration_n == 0:
        return 1500  # about this long for initial convergence
    else:
        return 10  # plenty for continuing finetuning


# use only for first epoch, to skip if already trained
def get_initial_checkpoints(instructions):
    checkpoints_to_load = []
    shard_dir = os.path.dirname(instructions.shards.config_save_loc)
    # temporary - use absolute file path
    if os.environ['DATA']:
        shard_dir = os.path.join(os.environ['DATA'], 'repos/zoobot', shard_dir)  # may not be necessary
    pretrained_checkpoints_dir = shard_dir + '_pretrained'
    if os.path.isdir(pretrained_checkpoints_dir):
        checkpoints_to_load = []
        # every folder inside might be a checkpoint, so add to list-of-checkpoints-to-load
        subdirs = os.listdir(pretrained_checkpoints_dir)
        for subdir in subdirs: # e.g. model_0, model_1 (does not include path) 
            possible_checkpoint_names = ['final', 'in_progress']
            for name in possible_checkpoint_names:
                checkpoint_loc = os.path.join(pretrained_checkpoints_dir, subdir, name)
                logging.info(f'Checking {checkpoint_loc}')
                if os.path.isfile(checkpoint_loc + '.index'):  # tf will always put this file in a ckpt directory
                    logging.info(f'Will load weights from {checkpoint_loc} for 0th epoch')
                    checkpoints_to_load.append(checkpoint_loc)
                break  # only add final, and fallback to try in_progress
        if len(checkpoints_to_load) == 0:
            raise FileNotFoundError(
                f'No checkpoints {possible_checkpoint_names} found in {pretrained_checkpoints_dir} under subdirs {subdirs}'
            )
    else:
        logging.warning(f'No initial prediction checkpoint found in {pretrained_checkpoints_dir}, not attempting to load')
        checkpoints_to_load = [None, None] # TODO hardcoded as two models by default, for now
    return checkpoints_to_load


def main(instructions_dir, this_iteration_dir, previous_iteration_dir, questions, label_cols, test=False):
    
    instructions = create_instructions.load_instructions(instructions_dir)
    with open(instructions.shard_config_loc, 'r') as f:
        shard_img_size = json.load(f)['size']
    assert instructions.ready()


    if os.path.isdir('/home/walml'):
        batch_size = 10 # 10 is the minimum possible size. 
    else:  # hopefully v100
        if os.environ['ZOOBOT_TEST'] == 'true':  # poor coding
            batch_size = 10
        else:
            batch_size = 128
    #  batch_norm etc fail explictly below 10, and maybe don't work so great until larger batch sizes. Paper (MnasNet) uses 4000!

    # for quick debugging - will not learn well locally
    if os.path.isdir('/home/walml'):
        final_size = 64 # 10 is the minimum possible size. 
    else:  # hopefully v100
        final_size = 224
    
    # TODO move this to create_instructions.py, replace train_callable
    fixed_estimator_params = run_estimator_config.FixedEstimatorParams(
        initial_size=shard_img_size,
        final_size=final_size,  # hardcode for now,
        crop_size=int(shard_img_size * 0.75),
        questions=questions,
        label_cols=label_cols,
        batch_size=batch_size 
    )

    acquisition_func = create_instructions.load_acquisition_func(instructions_dir).get()

    if test:  # override instructions and do a brief run only
        instructions.use_test_mode()

    oracle = oracles.load_oracle(instructions_dir)  # decoupled whether real or simulated
    initial_state = get_initial_state(instructions, this_iteration_dir, previous_iteration_dir)
    final_state = run(initial_state, instructions, fixed_estimator_params, acquisition_func, oracle, questions, label_cols, test=test)
    save_final_state(final_state, this_iteration_dir)

    # finally, tidy up by moving the log into the run directory
    # could not be create here because run directory did not exist at start of script
    if os.path.exists(log_loc):  # temporary workaround for disappearing log
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        shutil.move(log_loc, os.path.join(args.this_iteration_dir, '{}.log'.format(sha)))


if __name__ == '__main__':
    """
    python zoobot/active_learning/run_iteration.py --instructions-dir data/experiments/decals_multiq_sim/instructions --this-iteration-dir data/experiments/decals_multiq_sim/iteration_0 --previous-iteration-dir none  --test
    python zoobot/active_learning/run_iteration.py --instructions-dir data/experiments/decals_multiq_sim/instructions --this-iteration-dir data/experiments/decals_multiq_sim/iteration_1 --previous-iteration-dir data/experiments/decals_multiq_sim/iteration_0  --test
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')    
    if gpus:
        print('Using GPU: ', gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser(description='Execute single active learning iteration')
    parser.add_argument('--instructions-dir', dest='instructions_dir', type=str,
                    help='Directory with instructions for the execution of an iteration (e.g. shards_per_iter)')
    parser.add_argument('--this-iteration-dir', dest='this_iteration_dir', type=str,
                    help='Directory to save this iteration')
    parser.add_argument('--previous-iteration-dir', dest='previous_iteration_dir', type=str,
                    help='Directory with previously-executed iteration from which to begin (if provided)')
    parser.add_argument('--options', dest='options', default='')
    
    args = parser.parse_args()

    log_loc = 'run_iteration_{}.log'.format(time.time())

    logging.basicConfig(
        # filename=log_loc,
        # filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )
    # logging.getLogger().addHandler(logging.StreamHandler())

    questions = label_metadata.gz2_partial_questions
    label_cols = label_metadata.gz2_partial_label_cols
    # questions = label_metadata.gz2_questions
    # label_cols = label_metadata.gz2_label_cols

    options = args.options
    test = 'test' in options
    logging.info(f'Test mode: {test}')
    main(args.instructions_dir, args.this_iteration_dir, args.previous_iteration_dir, questions, label_cols, test)

