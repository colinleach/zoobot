"""Given a catalog, make:
- the shards
- the database
"""
import os
import shutil

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.active_learning import active_learning


def make_database_and_shards(catalog, db_loc, size, shard_dir, shard_size):
    if os.path.exists(db_loc):
        os.remove(db_loc)
    # set up db and shards using unknown catalog data
    db = active_learning.create_db(catalog, db_loc)
    columns_to_save = ['id_str']
    active_learning.write_catalog_to_tfrecord_shards(catalog, db, size, columns_to_save, shard_dir, shard_size=shard_size)
