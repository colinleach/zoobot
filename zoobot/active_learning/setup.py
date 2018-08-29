"""Given a catalog, make:
- the shards
- the database
"""
import os
import shutil

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.active_learning import active_learning


def make_initial_training_tfrecord(catalog, tfrecord_loc, size):
    """Save images with labels from `catalog` to tfrecord_loc
    Catalog is of images with known labels, to be used for initial training
    Save catalog to same location for stratify_probs to read 
    TODO messy csv saving
    
    Args:
        catalog (pd.DataFrame): with `id_str` (id string), `label` and `fits_loc` columns, to save
        tfrecord_loc (str): path to new tfrecord
        size (int): height/width dimension of image matrices
    """
    if os.path.exists(tfrecord_loc):
        os.remove(tfrecord_loc)
    # create initial small training set with random selection
    catalog_to_tfrecord.write_image_df_to_tfrecord(catalog, tfrecord_loc, size, ['id_str', 'label'])
    # save known catalog for stratify_probs to read
    catalog.to_csv(tfrecord_loc + '.csv')  


def make_database_and_shards(catalog, db_loc, size, shard_dir, shard_size):
    if os.path.exists(db_loc):
        os.remove(db_loc)
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.mkdir(shard_dir)
    # set up db and shards using unknown catalog data
    db = active_learning.create_db(catalog, db_loc)
    columns_to_save = ['id_str']
    active_learning.write_catalog_to_tfrecord_shards(catalog, db, size, columns_to_save, shard_dir, shard_size=shard_size)
