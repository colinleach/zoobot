"""Call this on a schedule to run active learning on Galaxy Zoo"""
import os
import time
import logging
import argparse

from zoobot.active_learning import create_instructions, run_iteration


def get_instructions_name(base_dir):
    return 'instructions_{}'.format(time.time())

def get_iteration_name(base_dir):
    return 'iteration_{}'.format(time.time())

def get_latest(items):
    """assumes items are identical except for unix timestamp at end, e.g. as above"""
    return sorted(items)[-1]  # largest timestamp


def iterate(base_dir, simulation, test):

    if simulation:
        logging.warning('Enabling simulation mode for this iteration')
        # TODO will need to toggle the mock_panoptes requests, currently always on
        # may need to pass this further forwards
        pass

    # if instructions have been made, use them. Otherwise, create them.
    paths_in_dir = os.listdir(base_dir)
    all_previous_instructions = [os.path.join(base_dir, x) for x in paths_in_dir if x.startswith('instructions_')]
    assert all_previous_instructions  # must be at least one instructions folder to use
    instructions_dir = get_latest(all_previous_instructions)

    all_previous_iterations = [os.path.join(base_dir, x) for x in paths_in_dir if x.startswith('iteration_')]
    if all_previous_iterations:  # continue from latest iteration
        previous_iteration_dir = get_latest(all_previous_iterations)
    else:  # start a new iteration
        previous_iteration_dir = None

    this_iteration_dir = get_iteration_name(base_dir)

    run_iteration.main(
        instructions_dir=instructions_dir,
        this_iteration_dir=this_iteration_dir,
        previous_iteration_dir=previous_iteration_dir,
        test=test
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Intelligently iterate to the next active learning step')
    parser.add_argument('--directory', dest='directory', type=str,
                    help='Base directory of active learning instructions and ')
    parser.add_argument('--instructions_dir', dest='instructions_dir', type=str,
                    help='Directory to save instructions')
    parser.add_argument('--simulation', dest='simulation', action='store_true', default=False,
                    help='Use previous classifications to simulate GZ, instead of truly uploading')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='Run only a minimal iteration as a functional test')
    args = parser.parse_args()

    log_loc = 'create_instructions_{}.log'.format(time.time())

    logging.basicConfig(
        filename=log_loc,
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    iterate(args.directory, args.simulation, args.test)

