
import pandas as pd
from zoobot.tfrecord import catalog_to_tfrecord

if __name__ == '__main__':

    for size in [128, 256, 512]:

        columns_to_save = ['t04_spiral_a08_spiral_count',
                           't04_spiral_a09_no_spiral_count',
                           't04_spiral_a08_spiral_weighted_fraction',
                           'id',
                           'ra',
                           'dec']

        # only exists if zoobot/get_catalogs/gz2 instructions have been followed
        df = pd.read_csv('/data/galaxy_zoo/gz2/subjects/all_labels_downloaded.csv',
                         usecols=columns_to_save + ['png_loc', 'png_ready'],
                         nrows=None)

        train_loc = 'data/gz2_spiral_{}_train.tfrecord'.format(size)
        test_loc = 'data/gz2_spiral_{}_test.tfrecord'.format(size)

        label_col = 't04_spiral_a08_spiral_weighted_fraction'

        catalog_to_tfrecord.write_catalog_to_train_test_tfrecords(
            df,
            label_col,
            train_loc,
            test_loc,
            size,
            columns_to_save)
