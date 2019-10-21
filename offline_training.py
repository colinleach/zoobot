  
import os
import argparse

from zoobot.active_learning import create_instructions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    parser.add_argument('--train-dir', dest='train_records_dir', type=str)
    parser.add_argument('--eval-dir', dest='eval_records_dir', type=str)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()

    shard_img_size = 256
    final_size = 128
    warm_start = False
    test = args.test
    epochs = args.epochs
    train_records_dir = args.train_records_dir
    eval_records_dir = args.eval_records_dir
    save_dir = args.save_dir
    train_records = [os.path.join(train_records_dir, x) for x in os.listdir(train_records_dir) if x.endswith('.tfrecord')]
    eval_records = [os.path.join(eval_records_dir, x) for x in os.listdir(eval_records_dir) if x.endswith('.tfrecord')]

    if test:
      batch_size = 4
    else:
      batch_size = 256

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
    train_callable(os.path.join(save_dir, 'results'), train_records, eval_records, learning_rate=0.01, epochs=epochs, batch_size=batch_size)  # can override default args here
