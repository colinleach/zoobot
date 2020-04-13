  
import os
import argparse

import tensorflow as tf

from zoobot.active_learning import create_instructions

  
if __name__ == '__main__':
    """
    Testing:

    python offline_training.py --experiment-dir results/latest_offline --shard-img-size 128 --train-dir data/decals/shards/multilabel_128/train --eval-dir data/decals/shards/multilabel_128/eval --epochs 2 
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
    final_size = int(shard_img_size / 2) # temp
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

    # parameters that only affect train_callable
    train_callable_obj = create_instructions.TrainCallableFactory(
        initial_size=shard_img_size,
        final_size=final_size,
        warm_start=warm_start,
        test=test,
    )
    train_callable_obj.save(save_dir)

    train_callable = train_callable_obj.get()
    questions = [
        'smooth-or-featured',
        'has-spiral-arms',
        'spiral-winding',
        'bar',
        'bulge-size'
    ]
    # network input x will eventually contain columns in this order
    label_cols = [
        'smooth-or-featured_smooth',
        'smooth-or-featured_featured-or-disk',
        'has-spiral-arms_yes',
        'has-spiral-arms_no',
        'spiral-winding_tight',
        'spiral-winding_medium',
        'spiral-winding_loose',
        'bar_strong',
        'bar_weak',
        'bar_no',
        'bulge-size_dominant',
        'bulge-size_large',
        'bulge-size_moderate',
        'bulge-size_small',
        'bulge-size_none'
    ]
     # can add to or override default args of train_callable here
    train_callable(os.path.join(save_dir, 'results'), train_records, eval_records, learning_rate=0.001, epochs=epochs, batch_size=batch_size, label_cols=label_cols, questions=questions) 
