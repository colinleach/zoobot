import pytest

import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.tests import TEST_EXAMPLE_DIR


def test_write_catalog_to_train_test_records(catalog, tfrecord_dir, size, columns_to_save):
    catalog_to_tfrecord.write_catalog_to_train_test_tfrecords(
        catalog,
        't04_spiral_a08_spiral_weighted_fraction',
        tfrecord_dir + '/train.tfrecords',
        tfrecord_dir + '/test.tfrecords',
        size,
        columns_to_save)
