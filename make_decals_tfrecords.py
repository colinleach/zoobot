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
        python make_decals_tfrecords.py --labelled-catalog=data/latest_labelled_catalog.csv --eval-size=2000 --shard-dir=data/decals/shards/multilabel_all_temp --img-size 128 --max 5000  --png-prefix /media/walml/beta/decals/png_native
        add --feat for filter

    Real:
        python make_decals_tfrecords.py --labelled-catalog=data/decals/decals_master_catalog.csv --eval-size=1000 --shard-dir=data/decals/shards/multilabel_master_filtered_64 --img-size 64 --png-prefix /media/walml/beta/decals/png_native --feat
        python make_decals_tfrecords.py --labelled-catalog=data/decals/decals_master_catalog.csv --eval-size=1000 --shard-dir=data/decals/shards/multilabel_master_filtered_128 --img-size 128 --png-prefix /media/walml/beta/decals/png_native --feat
        python make_decals_tfrecords.py --labelled-catalog=data/decals/decals_master_catalog.csv --eval-size=1000 --shard-dir=data/decals/shards/multilabel_master__filtered_256 --img-size 256 --png-prefix /media/walml/beta/decals/png_native  --feat

    And for GZ2:

        python make_decals_tfrecords.py --labelled-catalog=data/gz2/gz2_master_catalog.csv --eval-size=1000 --shard-dir=data/gz2/shards/multilabel_master_filtered_64 --img-size 64 --png-prefix /media/walml/beta/gz2/png --feat
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
    parser.add_argument('--feat', dest='featured_filter', action='store_true', default=False, help='Apply filter to featured face-on. Not quite identical to usual shards.')

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

    # copied from define_experiment.py
    is_retired = (36 < labelled_catalog['smooth-or-featured_total-votes']) & (labelled_catalog['smooth-or-featured_total-votes'] < 45)
    labelled_catalog = labelled_catalog[is_retired]
    if args.featured_filter:
        print(labelled_catalog.columns.values)
        len_before = len(labelled_catalog)
        min_featured = 0.5  # will be a bit different, volunteers here
        is_featured = (labelled_catalog['smooth-or-featured_featured-or-disk'] / labelled_catalog['smooth-or-featured_total-votes']) > min_featured
        featured = labelled_catalog[is_featured]
        is_face_on = (featured['disk-edge-on_yes'] / featured['smooth-or-featured_featured-or-disk']) < 0.5
        labelled_catalog = featured[is_face_on]
        logging.info(f'{len_before} before filter, {len(labelled_catalog)} after filter')
        print(f'{len_before} before filter, {len(labelled_catalog)} after filter')

    labelled_catalog = labelled_catalog.sample(
        min(len(labelled_catalog), max_labelled))  # shuffle and cut (if needed)
    assert len(labelled_catalog) > 0

    for col in label_cols:
        labelled_catalog[col] = labelled_catalog[col].astype(float)
    # remember that if a catalog has both png_loc and file_loc, it will read png_loc
    if args.png_prefix != '':
        if 'local_png_loc' in labelled_catalog.columns.values:
            labelled_catalog['file_loc'] = labelled_catalog['local_png_loc'].apply(lambda x: args.png_prefix + x[32:])
            del labelled_catalog['local_png_loc']
        else:
            labelled_catalog['file_loc'] = labelled_catalog['png_loc'].apply(lambda x: args.png_prefix + x[32:])
            del labelled_catalog['png_loc']
   
    assert 'file_loc' in labelled_catalog.columns.values
    logging.info('Expected files at: {}'.format(labelled_catalog['file_loc'].iloc[0]))

    if 'id_str' not in labelled_catalog.columns.values:
        labelled_catalog['id_str'] = labelled_catalog['iauname'].astype(str)

    train_test_fraction = catalog_to_tfrecord.get_train_test_fraction(
        len(labelled_catalog), eval_size)
    train_dir = os.path.join(shard_dir, 'train')
    eval_dir = os.path.join(shard_dir, 'eval')
    for directory in [shard_dir, train_dir, eval_dir]:  # order matters
        if not os.path.exists(directory):
            os.mkdir(directory)

    # TEMPORARY throw a warning and add fake labels. Useful for master catalog
    # logging.critical('Adding fake labels!')
    # for label_col in label_cols:
    #     labelled_catalog[label_col] = 0.

    train_df, eval_df = catalog_to_tfrecord.split_df(
        labelled_catalog, train_test_fraction=train_test_fraction)
    logging.info('\nTraining subjects: {}'.format(len(train_df)))
    logging.info('Eval subjects: {}'.format(len(eval_df)))
    if len(train_df) < len(eval_df):
        print('More eval subjects than training subjects - is this intended?')
    # exit()

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
