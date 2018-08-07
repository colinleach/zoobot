import pytest

from zoobot import read_tfrecord


def test_load_examples_from_tfrecord(example_tfrecord):
    tfrecord_locs = [example_tfrecord['loc']]
    examples = read_tfrecord.load_examples_from_tfrecord([tfrecord_loc], 1, size, channels)
    # TODO asserts


def test_show_example()