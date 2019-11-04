import tensorflow as tf


def simple_shuffle_batch(source, capacity, batch_size=10):
  # Create a random shuffle queue.
  queue = tf.RandomShuffleQueue(capacity=capacity,
                                min_after_dequeue=int(0.9*capacity),
                                shapes=source.shape, dtypes=source.dtype)

  # Create an op to enqueue one item.
  enqueue = queue.enqueue(source)

  # Create a queue runner that, when started, will launch 4 threads applying
  # that enqueue op.
  num_threads = 4
  qr = tf.train.QueueRunner(queue, [enqueue] * num_threads)

  # Register the queue runner so it can be found and started by
  # `tf.train.start_queue_runners` later (the threads are not launched yet).
  tf.train.add_queue_runner(qr)

  # Create an op to dequeue a batch
  return queue.dequeue_many(batch_size)


# create a dataset that counts from 0 to 99
input = tf.constant(list(range(100)))
input = tf.data.Dataset.from_tensor_slices(input)
input = input.make_one_shot_iterator().get_next()

# Create a slightly shuffled batch from the sorted elements
get_batch = simple_shuffle_batch(input, capacity=20)

# `MonitoredSession` will start and manage the `QueueRunner` threads.
with tf.train.MonitoredSession() as sess:
  # Since the `QueueRunners` have been started, data is available in the
  # queue, so the `sess.run(get_batch)` call will not hang.
  while not sess.should_stop():
    print(sess.run(get_batch))
