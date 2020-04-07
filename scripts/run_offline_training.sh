# Requires DATA_DIR env variable

export PNG_PREFIX=$DATA_DIR
export SHARD_IMG_SIZE=256
export SHARD_MAX=''
export SHARD_EVAL='2500'

# local:
export PYTHON=python
export DATA_DIR=/home/repos/zoobot/data
export PNG_PREFIX=/Volumes/alpha
export SHARD_IMG_SIZE=64
export SHARD_MAX='--max=1000'
export SHARD_EVAL='500'

# shared
export SHARD_DIR=$DATA_DIR/decals/shards/multilabel_feat10_$SHARD_IMG_SIZE
export LABELLED_CATALOG=$DATA_DIR/prepared_catalogs/mac_catalog_feat10_correct_labels_full_256.csv
export EXPERIMENT_DIR=$DATA_DIR/experiments/multiquestion_smooth_spiral_feat10_tf2_$SHARD_IMG_SIZE
export EPOCHS=2

export BRANCH=tf2
cd zoobot && git pull && git checkout $BRANCH && cd ../

# make test shards (if needed)
python zoobot/make_decals_tfrecords.py --labelled-catalog $DATA_DIR/decals/prepared_catalogs/decals_smooth_may/labelled_catalog.csv  --shard-dir=$SHARD_DIR --img-size=$SHARD_IMG_SIZE --eval-size=$SHARD_EVAL $SHARD_MAX --png-prefix=$PNG_PREFIX

# run training
python zoobot/offline_training.py --experiment-dir $EXPERIMENT_DIR --train-dir $SHARD_DIR/train --eval-dir $SHARD_DIR/eval --shard-img-size=$SHARD_IMG_SIZE --epochs $EPOCHS