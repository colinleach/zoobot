import os
import logging
import argparse

import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.active_learning import database


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make shards')
    parser.add_argument('--labelled-catalog', dest='labelled_catalog_loc', type=str,
                        help='Path to csv catalog of previous labels and file_loc, for shards')
    parser.add_argument('--eval-size', dest='eval_size', type=str,
                        help='Path to csv catalog of previous labels and file_loc, for shards')
    parser.add_argument('--shard-dir', dest='shard_dir', type=str,
                        help='Directory into which to place shard directory')
    parser.add_argument('--img-size', dest='img_size', type=int, default=256,
                        help='Directory into which to place shard directory')
    parser.add_argument('--max', dest='max_labelled', type=int, default=10000000000,
                        help='Max galaxies (for debugging/speed')
    parser.add_argument('--png-prefix', dest='png_prefix', type=str, default='')


    args = parser.parse_args()
    labelled_catalog_loc = args.labelled_catalog_loc
    max_labelled = args.max_labelled
    shard_dir = args.shard_dir
    eval_size = args.eval_size
    img_size = args.img_size
    shard_size = 4096

    labelled_catalog = pd.read_csv(labelled_catalog_loc)
    labelled_catalog = labelled_catalog.sample(
        min(len(labelled_catalog), max_labelled))  # shuffle and cut (if needed)
    train_test_fraction = catalog_to_tfrecord.get_train_test_fraction(
        len(labelled_catalog), eval_size)
    train_dir = os.path.join(shard_dir, 'train')
    eval_dir = os.path.join(shard_dir, 'eval')
    for directory in [shard_dir, train_dir, eval_dir]:  # order matters
        if not os.path.exists(directory):
            os.mkdir(directory)

    assert len(labelled_catalog) > 0
    if args.png_prefix != '':
        labelled_catalog['file_loc'] = labelled_catalog['file_loc'].apply(lambda x: args.png_prefix + x[35:])
        logging.info('Expected files at: {}'.format(labelled_catalog['file_loc'].iloc[0]))

    train_df, eval_df = catalog_to_tfrecord.split_df(
        labelled_catalog, train_test_fraction=train_test_fraction)
    logging.info('\nTraining subjects: {}'.format(len(train_df)))
    logging.info('Eval subjects: {}'.format(len(eval_df)))
    if len(train_df) < len(eval_df):
        print('More eval subjects than training subjects - is this intended?')

    label_cols = ['smooth-or-featured_smooth', 'smooth-or-featured_featured-or-disk', 'smooth-or-featured_artifact', 'smooth-or-featured_total-votes', 'bar_strong', 'bar_weak', 'bar_no', 'bar_total-votes',  'spiral_yes', 'spiral_no', 'spiral_total-votes']
    columns_to_save = ['id_str'] + label_cols
    for (df, save_dir) in [(train_df, train_dir), (eval_df, eval_dir)]:
        database.write_catalog_to_tfrecord_shards(
            df,
            db=None,
            img_size=img_size,
            columns_to_save=columns_to_save,  # TODO use schema to save all?
            save_dir=save_dir,
            shard_size=shard_size
        )
