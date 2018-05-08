import pytest

import pandas as pd

from zoobot.tfrecord import gz2_to_tfrecord

TEST_EXAMPLE_DIR = 'zoobot/test_examples'


@pytest.fixture
def record_dir(tmpdir):
    return tmpdir.mkdir('record_dir').strpath


@pytest.fixture
def columns_to_save():
    return [
        't04_spiral_a08_spiral_count',
        't04_spiral_a09_no_spiral_count',
        't04_spiral_a08_spiral_weighted_fraction',
        # 'id',  Note: saving string features is not yet implemented
        'ra',
        'dec']


@pytest.fixture
def downloaded_catalog():

    zoo1 = {
        't04_spiral_a08_spiral_count': 11,
        't04_spiral_a09_no_spiral_count': 3,
        't04_spiral_a08_spiral_weighted_fraction': 0.4,
        'id': 'zoo1',
        'ra': 12.0,
        'dec': -1.0,
        'png_loc': '{}/example_a.png'.format(TEST_EXAMPLE_DIR),
        'png_ready': True
    }

    zoo2 = {
        't04_spiral_a08_spiral_count': 11,
        't04_spiral_a09_no_spiral_count': 3,
        't04_spiral_a08_spiral_weighted_fraction': 0.4,
        'id': 'zoo1',
        'ra': 15.0,
        'dec': -1.0,
        'png_loc': '{}/example_b.png'.format(TEST_EXAMPLE_DIR),
        'png_ready': True
    }

    return pd.DataFrame([zoo1, zoo2] * 128)  # 256 examples


def test_write_catalog_to_train_test_records(downloaded_catalog, record_dir, columns_to_save):
    gz2_to_tfrecord.write_catalog_to_train_test_tfrecords(
        downloaded_catalog,
        't04_spiral_a08_spiral_weighted_fraction',
        record_dir + '/train.tfrecords',
        record_dir + '/test.tfrecords',
        32,
        columns_to_save)
