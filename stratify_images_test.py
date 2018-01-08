import tensorflow as tf
import numpy as np

from input_utils import stratify_images

"""
Computational graph
"""

SIZE = 4
true_image_values = 3.
false_image_values = true_image_values * -1.

n_true_examples = 1
n_false_examples = 3

batch = 2  # keeping it simple for now

true_images = tf.Variable(initial_value=np.ones((n_true_examples, SIZE, SIZE, 1)) * true_image_values, dtype=tf.float32)
false_images = tf.Variable(initial_value=np.ones((n_false_examples, SIZE, SIZE, 1)) * false_image_values, dtype=tf.float32)

true_labels = tf.Variable(initial_value=np.ones(n_true_examples), dtype=tf.int32)
false_labels = tf.Variable(initial_value=np.zeros(n_false_examples), dtype=tf.int32)


batch_images = tf.concat([true_images, false_images], 0)
batch_labels = tf.concat([true_labels, false_labels], 0)

strat_images, strat_labels = stratify_images(batch_images, batch_labels, batch)

"""
Execution
"""

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

true_images, false_images, true_labels, false_labels = sess.run([true_images, false_images, true_labels, false_labels])

# print(true_images)
# print(false_images)
# print(true_labels)
# print(false_labels)

batch_images, batch_labels = sess.run([batch_images, batch_labels])

# print(batch_images)
# print(batch_labels)

strat_images, strat_labels = sess.run([strat_images, strat_labels])

print(strat_images)
print(strat_labels)

print(strat_images.shape)
print(strat_labels.shape)

assert np.sum(strat_labels) == int(batch/2)
assert np.mean(strat_images) == 0.0

hopefully_true_images = strat_images[strat_labels == 1]
assert np.min(hopefully_true_images) == true_image_values
