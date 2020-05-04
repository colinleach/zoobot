  
import os
import argparse
import time

import tensorflow as tf

from zoobot.active_learning import create_instructions, run_estimator_config
from zoobot.estimators import schema

  
if __name__ == '__main__':
    """

      python offline_training.py --experiment-dir results/latest_offline_retired --shard-img-size 128 --train-dir data/decals/shards/multilabel_128_retired/train --eval-dir data/decals/shards/multilabel_128_retired/eval --epochs 150 
      
      To make model for smooth/featured (also change cols below):
      python offline_training.py --experiment-dir results/smooth_or_featured_offline --shard-img-size 128 --train-dir data/decals/shards/multilabel_128/train --eval-dir data/decals/shards/multilabel_128/eval --epochs 150 

    Testing:
      python offline_training.py --experiment-dir results/debug --shard-img-size 128 --train-dir data/decals/shards/multilabel_128/train --eval-dir data/decals/shards/multilabel_128/eval --epochs 2 
    """

    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

    # check which GPU we're using, helpful on ARC
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("GPUs:",  physical_devices)

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    parser.add_argument('--shard-img-size', dest='shard_img_size', type=int, default=256)
    parser.add_argument('--train-dir', dest='train_records_dir', type=str)
    parser.add_argument('--eval-dir', dest='eval_records_dir', type=str)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--warm-start', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()

    shard_img_size = args.shard_img_size
    # final_size = int(shard_img_size / 2) # temp
    final_size = 64
    warm_start = args.warm_start
    test = args.test
    epochs = args.epochs
    train_records_dir = args.train_records_dir
    eval_records_dir = args.eval_records_dir
    save_dir = args.save_dir
    train_records = [os.path.join(train_records_dir, x) for x in os.listdir(train_records_dir) if x.endswith('.tfrecord')]
    eval_records = [os.path.join(eval_records_dir, x) for x in os.listdir(eval_records_dir) if x.endswith('.tfrecord')]

    if test:
      batch_size = 32
    else:
      batch_size = 64  # small for now for laptop GPU, was 256

    if not os.path.isdir(save_dir):
      os.mkdir(save_dir)

    # must match label cols below
    questions = [
        'smooth-or-featured',
        # 'has-spiral-arms',
        # 'spiral-winding',
        # 'bar',
        # 'bulge-size'
    ]

    # will load labels from shard, in this order
    # will predict all label columns, in this order
    label_cols = [
        'smooth-or-featured_smooth',
        'smooth-or-featured_featured-or-disk',
        # 'has-spiral-arms_yes',
        # 'has-spiral-arms_no',
        # 'spiral-winding_tight',
        # 'spiral-winding_medium',
        # 'spiral-winding_loose',
        # 'bar_strong',
        # 'bar_weak',
        # 'bar_no',
        # 'bulge-size_dominant',
        # 'bulge-size_large',
        # 'bulge-size_moderate',
        # 'bulge-size_small',
        # 'bulge-size_none'
    ]
    schema = losses.Schema(label_cols, questions, version='decals')

    run_config = run_estimator_config.get_run_config(
      initial_size=shard_img_size,
      final_size=final_size,
      warm_start=warm_start,
      log_dir=save_dir,
      train_records=train_records,
      eval_records=eval_records,
      epochs=epochs,
      schema=schema
      batch_size=batch_size
    )
    
    run_config.run_estimator() 
