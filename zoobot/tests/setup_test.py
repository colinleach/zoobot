import os

import pytest
import sqlite3

from zoobot.active_learning import setup
from zoobot.tests.active_learning_test import verify_db_matches_shards, verify_catalog_matches_shards


def test_setup_fixed_images(catalog, db_loc, size, channels, tfrecord_dir):
    assert os.path.isdir(tfrecord_dir)
    setup.make_database_and_shards(catalog, db_loc, size, tfrecord_dir, shard_size=25)
    db = sqlite3.connect(db_loc)
    # verify_db_matches_catalog(catalog, db)
    verify_db_matches_shards(db, size, channels)
    verify_catalog_matches_shards(catalog, db, size, channels)


# TODO use proper fixture factory instead of just duplicating
# def test_setup_fixed_images(catalog_random_images, db_loc, size, channels, tfrecord_dir):
#     catalog = catalog_random_images
#     assert os.path.isdir(tfrecord_dir)
#     setup.make_database_and_shards(catalog, db_loc, size, tfrecord_dir, shard_size=25)
#     db = sqlite3.connect(db_loc)
#     # verify_db_matches_catalog(catalog, db)
#     verify_db_matches_shards(db, size, channels)
#     verify_catalog_matches_shards(catalog, db, size, channels)

