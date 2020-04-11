import os
import logging
import argparse

import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.active_learning import database
from zoobot.tests.estimators.input_utils_test import label_cols
from gzreduction.deprecated import dr5_schema  # not deprecated any more...


if __name__ == '__main__':
    """

    Labelled catalog must include id_str (aka iauname) and png_loc as well as any desired label columns
    Testing:
        python make_decals_tfrecords.py --labelled-catalog=data/latest_labelled_catalog.csv --eval-size=2000 --shard-dir=data/decals/shards/multilabel_128 --img-size 128 --max 5000  --png-prefix /media/walml/beta/decals/png_native

    """

    logging.basicConfig(
        format='%(levelname)s:%(message)s',
        level=logging.INFO)


    parser = argparse.ArgumentParser(description='Make shards')
    parser.add_argument('--labelled-catalog', dest='labelled_catalog_loc', type=str,
                        help='Path to csv catalog of previous labels and file_loc, for shards')
    parser.add_argument('--eval-size', dest='eval_size', type=int,
                        help='Path to csv catalog of previous labels and file_loc, for shards')
    parser.add_argument('--shard-dir', dest='shard_dir', type=str,
                        help='Directory into which to place shard directory')
    parser.add_argument('--img-size', dest='img_size', type=int, default=256,
                        help='Directory into which to place shard directory')
    parser.add_argument('--max', dest='max_labelled', type=int, default=10000000000,
                        help='Max galaxies (for debugging/speed')
    parser.add_argument('--png-prefix', dest='png_prefix', type=str, default='', help='prefix to use before dr5/J00, replacing any existing prefix')

    # order is not important here, just keying the serialise_example dict, but it matters in offline_training.py
    label_cols = [
        'smooth-or-featured_smooth',
        'smooth-or-featured_featured-or-disk',
        'has-spiral-arms_yes',
        'has-spiral-arms_no',
        'bar_strong',
        'bar_weak',
        'bar_no',
        'bulge-size_dominant',
        'bulge-size_large',
        'bulge-size_moderate',
        'bulge-size_small',
        'bulge-size_none',
        'spiral-winding_tight',
        'spiral-winding_medium',
        'spiral-winding_loose'
    ]

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
    assert len(labelled_catalog) > 0

    for col in label_cols:
        labelled_catalog[col] = labelled_catalog[col].astype(float)
    # remember that if a catalog has both png_loc and file_loc, it will read png_loc
    if args.png_prefix != '':
        labelled_catalog['file_loc'] = labelled_catalog['png_loc'].apply(lambda x: args.png_prefix + x[32:])
        del labelled_catalog['png_loc']
        logging.info('Expected files at: {}'.format(labelled_catalog['file_loc'].iloc[0]))
    assert 'file_loc' in labelled_catalog.columns.values

    if 'id_str' not in labelled_catalog.columns.values:
        labelled_catalog['id_str'] = labelled_catalog['iauname'].astype(str)

    train_test_fraction = catalog_to_tfrecord.get_train_test_fraction(
        len(labelled_catalog), eval_size)
    train_dir = os.path.join(shard_dir, 'train')
    eval_dir = os.path.join(shard_dir, 'eval')
    for directory in [shard_dir, train_dir, eval_dir]:  # order matters
        if not os.path.exists(directory):
            os.mkdir(directory)

    train_df, eval_df = catalog_to_tfrecord.split_df(
        labelled_catalog, train_test_fraction=train_test_fraction)
    logging.info('\nTraining subjects: {}'.format(len(train_df)))
    logging.info('Eval subjects: {}'.format(len(eval_df)))
    if len(train_df) < len(eval_df):
        print('More eval subjects than training subjects - is this intended?')

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
