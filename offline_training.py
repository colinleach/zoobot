  
import os
import argparse
import time
import logging

import tensorflow as tf

from zoobot.active_learning import create_instructions, run_estimator_config
from zoobot.estimators import losses, input_utils

  
if __name__ == '__main__':
    """
    To make model for smooth/featured (also change cols below):
      # python offline_training.py --experiment-dir results/smooth_or_featured_offline --shard-img-size 128 --train-dir data/decals/shards/multilabel_master_filtered_128/train --eval-dir data/decals/shards/multilabel_master_filtered_128/eval --epochs 1000 
      python offline_training.py --experiment-dir results/smooth_or_featured_offline --shard-img-size 256 --train-dir data/decals/shards/multilabel_master_filtered_256/train --eval-dir data/decals/shards/multilabel_master_filtered_256/eval --epochs 1000 --batch-size 8 --final-size 128

    To make model for predictions on all cols, for appropriate galaxies only:
      python offline_training.py --experiment-dir results/latest_offline_featured --shard-img-size 128 --train-dir data/decals/shards/multilabel_master_filtered_128/train --eval-dir data/decals/shards/multilabel_master_filtered_128/eval --epochs 1000 
      

    GZ2 testing:
      python offline_training.py --experiment-dir results/debug --shard-img-size 256 --train-dir data/gz2/shards/all_featp5_facep5_sim_256/train_shards --eval-dir data/gz2/shards/all_featp5_facep5_sim_256/eval_shards --epochs 2 --batch-size 8 --final-size 128

    Local testing:
      python offline_training.py --experiment-dir results/debug --shard-img-size 128 --train-dir data/decals/shards/multilabel_master_filtered_128/train --eval-dir data/decals/shards/multilabel_master_filtered_128/eval --epochs 2 --batch-size 16
      
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
    parser.add_argument('--final-size', dest='final_size', type=int, default=64)
    parser.add_argument('--train-dir', dest='train_records_dir', type=str)
    parser.add_argument('--eval-dir', dest='eval_records_dir', type=str)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--batch-size', dest='batch_size', default=64, type=int)
    parser.add_argument('--warm-start', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
      format='%(levelname)s:%(message)s',
      level=logging.INFO)

    shard_img_size = args.shard_img_size
    final_size = args.final_size  # step time prop. to resolution
    batch_size = args.batch_size
    logging.info('Batch {}, final size {}'.format(batch_size, final_size))
    warm_start = args.warm_start
    test = args.test
    epochs = args.epochs
    train_records_dir = args.train_records_dir
    eval_records_dir = args.eval_records_dir
    save_dir = args.save_dir
    train_records = [os.path.join(train_records_dir, x) for x in os.listdir(train_records_dir) if x.endswith('.tfrecord')]
    eval_records = [os.path.join(eval_records_dir, x) for x in os.listdir(eval_records_dir) if x.endswith('.tfrecord')]

    if not os.path.isdir(save_dir):
      os.mkdir(save_dir)

    # must match label cols below
    questions = [
        'smooth-or-featured',
        'has-spiral-arms',
        'bar',
        'bulge-size'
    ]

    # will load labels from shard, in this order
    # will predict all label columns, in this order
    if 'decals' in train_records_dir:
        version='decals'
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
            'bulge-size_none'
        ]
    else:
        version='gz2'
        # gz2 cols
        label_cols = [
            'smooth-or-featured_smooth',
            'smooth-or-featured_featured-or-disk',
            'has-spiral-arms_yes',
            'has-spiral-arms_no',
            'bar_yes',
            'bar_no',
            'bulge-size_dominant',
            'bulge-size_obvious',
            'bulge-size_just-noticeable',
            'bulge-size_no'
        ]
    schema = losses.Schema(label_cols, questions, version=version)

    print('Epochs: {}'.format(epochs))

    run_config = run_estimator_config.get_run_config(
      initial_size=shard_img_size,
      final_size=final_size,
      crop_size=int(shard_img_size * 0.75),
      warm_start=warm_start,
      log_dir=save_dir,
      train_records=train_records,
      eval_records=eval_records,
      epochs=epochs,
      schema=schema,
      batch_size=batch_size,
      patience=5
    )

    # check for bad shard_img_size leading to bad batch size
    # train_dataset = input_utils.get_input(config=run_config.train_config)
    # test_dataset = input_utils.get_input(config=run_config.eval_config)
    # for x, y in train_dataset.take(2):
    #     print(x.shape, y.shape)
    #     assert x.shape[0] == batch_size
    
    trained_model = run_config.run_estimator() 
