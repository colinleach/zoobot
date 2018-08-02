import tensorflow as tf

"""
Computational graph
"""


SIZE = 4
true_image_values = 3.
false_image_values = true_image_values * -1.

n_true_examples = 40
n_false_examples = 120

batch_size = 8  # keeping it simple for now

data = tf.constant([1. for n in range(90)] + [0. for n in range(10)])
labels = tf.constant([1 for n in range(90)] + [0 for n in range(10)])
dataset = tf.data.Dataset.from_tensor_slices(
    {'data': data,
     'labels': labels}
)

dataset = dataset.shuffle(buffer_size=10000)
# dataset = dataset.batch(32)
dataset = dataset.repeat(10)
result = dataset.make_one_shot_iterator().get_next()


"""
Execution
"""

with tf.train.MonitoredSession() as sess:
    print(sess.run(result))
    print(sess.run(result))
    print(sess.run(result))
    print(sess.run(result))
    while not sess.should_stop():
        print(sess.run(result))
