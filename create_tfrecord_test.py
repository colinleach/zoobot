import pytest

import os

from create_tfrecord import *


@pytest.fixture()
def example_image_data():
    return np.array(np.ones((50, 50, 3)))


@pytest.fixture
def tfrecord_dir(tmpdir):
    return tmpdir.mkdir('tfrecord_dir').strpath


def test_matrix_to_tfrecord(example_image_data, tfrecord_dir):
    label = 1
    save_loc = '{}/example.tfrecords'.format(tfrecord_dir)
    assert not os.path.exists(save_loc)
    image_to_tfrecord(example_image_data, label, save_loc)
    assert os.path.exists(save_loc)


def test_matrix_to_tfrecord_with_extra_data(example_image_data, tfrecord_dir):
    label = 1
    save_loc = '{}/example.tfrecords'.format(tfrecord_dir)
    extra_data = {
        'an_int': 1,
        'a_float': 2.,
        'some_floats': np.array([1., 2., 3.])
    }
    assert not os.path.exists(save_loc)
    image_to_tfrecord(example_image_data, label, save_loc, extra_data=extra_data)
    assert os.path.exists(save_loc)
