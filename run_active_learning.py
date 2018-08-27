import os
import shutil
import logging

import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator
from zoobot.active_learning import estimator_from_disk, active_learning
from zoobot.tests import TEST_EXAMPLE_DIR

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(
    filename='run_active_learning.log',
    filemode='w',
    format='%(asctime)s %(message)s',
    level=logging.DEBUG)

db_loc = 'active_learning.db'
id_col = 'id_str'
label_col = 'label'
label_split_value = '0.4'
initial_size = 64
final_size = 28
channels = 3
predictor_dir = os.path.join(CURRENT_DIR, 'zoobot/runs/active')
shard_dir = os.path.join(CURRENT_DIR, 'zoobot/data/shards')
shard_size = 25

train_tfrecord_loc = os.path.join(CURRENT_DIR, 'zoobot/data/panoptes_featured_s{}_l{}_active_train.tfrecord').format(  # empty
    initial_size, 
    label_split_value
)
eval_tfrecord_loc = os.path.join(CURRENT_DIR, 'zoobot/data/panoptes_featured_s{}_l{}_test.tfrecord').format(
    initial_size, 
    label_split_value
)
assert os.path.exists(eval_tfrecord_loc)

run_name = 'active_si{}_sf{}_l{}'.format(initial_size, final_size, label_split_value)

# train_tf_loc = os.path.join(CURRENT_DIR, 'zoobot/data/panoptes_featured_s28_l0.4_train.tfrecord.csv'
# test_tf_loc = os.path.join(CURRENT_DIR, 'zoobot/data/panoptes_featured_s28_l0.4_test.tfrecord.csv'

train_callable = lambda: run_estimator.run_estimator(run_config)

# new database (for now)
if os.path.exists(db_loc):
    os.remove(db_loc)

# new active learning train tfrecord (for now)
if os.path.exists(train_tfrecord_loc):
    os.remove(train_tfrecord_loc)

# new shards always (for now)
if os.path.exists(shard_dir):
    shutil.rmtree(shard_dir)
os.mkdir(shard_dir)

# new predictors (apart from initial disk load) for now
if os.path.exists(predictor_dir):
    shutil.rmtree(predictor_dir)
os.mkdir(predictor_dir)


catalog = pd.read_csv(os.path.join(CURRENT_DIR, 'zoobot/data/panoptes_featured_s{}_l{}_train.tfrecord.csv').format(initial_size, label_split_value))

# >36 votes required, gives low count uncertainty
catalog = catalog[catalog['smooth-or-featured_total-votes'] > 36]  
# make labels
catalog[label_col] = (catalog['smooth-or-featured_smooth_fraction'] > float(label_split_value)).astype(int)  # 0 for featured
# make ids
catalog[id_col] = catalog['subject_id'].astype(str) 

known_catalog = catalog[:50]  # for initial training data
unknown_catalog = catalog[50:100]  # for new data
unknown_catalog.to_csv(os.path.join(TEST_EXAMPLE_DIR, 'panoptes.csv'))

# create initial small training set with random selection
catalog_to_tfrecord.write_image_df_to_tfrecord(known_catalog, train_tfrecord_loc, initial_size, [id_col, label_col])
known_catalog.to_csv(train_tfrecord_loc + '.csv')  # save known catalog for stratify_probs to read
# define the estimator
run_config = estimator_from_disk.setup(run_name, train_tfrecord_loc, eval_tfrecord_loc, initial_size, final_size, label_split_value, predictor_dir)
# set up db and shards using unknown catalog data
active_learning.setup(unknown_catalog, db_loc, id_col, label_col, initial_size, shard_dir, shard_size)
# run active learning (labels currently not implemented)
active_learning.run(unknown_catalog, db_loc, id_col, label_col, initial_size, channels, predictor_dir, train_tfrecord_loc, train_callable)
