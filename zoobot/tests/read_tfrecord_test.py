import pytest

from zoobot import read_tfrecord


def test_load_examples_from_tfrecord():
    examples = read_tfrecord.load_examples_from_tfrecord([tfrecord_loc], 1, size, channels)
    # TODO asserts


def 