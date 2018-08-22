import os

import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zoobot.tests import TEST_EXAMPLE_DIR
from zoobot.tfrecord import read_tfrecord


def test_load_examples_from_tfrecord(example_tfrecord_loc, size, channels):
    tfrecord_locs = [example_tfrecord_loc]
    examples = read_tfrecord.load_examples_from_tfrecord(tfrecord_locs, size, channels, 5)
    assert len(examples) == 5
    example = examples[0]
    assert 0. < example['matrix'].mean() < 255.
    assert example['label'] == 1 or example['label'] == 0
    plt.clf()
    plt.imshow(example['matrix'].reshape(size, size, channels))
    plt.savefig(os.path.join(TEST_EXAMPLE_DIR, 'loaded_image_from_example_tfrecord.png'))



def test_load_examples_from_tfrecord_all(example_tfrecord_loc, size, channels):
    tfrecord_locs = [example_tfrecord_loc]
    examples = read_tfrecord.load_examples_from_tfrecord(tfrecord_locs, size, channels, None)
    assert len(examples) > 5
    example = examples[0]
    assert 0. < example['matrix'].mean() < 255.
    assert example['label'] == 1 or example['label'] == 0
    plt.clf()
    plt.imshow(example['matrix'].reshape(size, size, channels))
    plt.savefig(os.path.join(TEST_EXAMPLE_DIR, 'loaded_image_from_example_tfrecord.png'))


def test_show_example(parsed_example, size, channels):
    fig, ax = plt.subplots()
    read_tfrecord.show_example(parsed_example, size, channels, ax)
    fig.savefig(os.path.join(TEST_EXAMPLE_DIR, 'show_example_for_visual_check_image.png'))


def test_show_examples(parsed_example, size, channels):
    fig, axes = read_tfrecord.show_examples([parsed_example for n in range(6)], size, channels)
    fig.savefig(os.path.join(TEST_EXAMPLE_DIR, 'show_examples_for_many_visual_check_image.png'))
