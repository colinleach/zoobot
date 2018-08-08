import os

import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zoobot.tests import TEST_EXAMPLE_DIR
from zoobot.tfrecord import read_tfrecord


def test_load_examples_from_tfrecord(example_tfrecord_loc, size, channels):
    tfrecord_locs = [example_tfrecord_loc]
    examples = read_tfrecord.load_examples_from_tfrecord(tfrecord_locs, 5, size, channels)
    # TODO asserts
    plt.clf()
    plt.imshow(examples[0]['matrix'].reshape(size, size, channels))
    plt.savefig(os.path.join(TEST_EXAMPLE_DIR, 'original_minimal_loaded_image.png'))


def test_show_example():
    pass  # TODO
