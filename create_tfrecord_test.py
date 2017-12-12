import pytest

from create_tfrecord import *


@pytest.fixture()
def example_image_data():
    return np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])


def test_array_to_tfrecord(example_image_data):
    array_to_tfrecord(example_image_data, 'example.tfrecords')
