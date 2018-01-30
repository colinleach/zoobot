import os
import random

import numpy as np
import tensorflow as tf

from zoobot.models.input_utils import input
from zoobot.catalogs.tfrecord.create_tfrecord import image_to_tfrecord

"""
Make test data
"""

size = 4
true_image_values = 3.
false_image_values = true_image_values * -1.

n_true_examples = 200
n_false_examples = 800

true_images = [np.ones((size, size, 3), dtype=float) * true_image_values for n in range(n_true_examples)]
false_images = [np.ones((size, size, 3), dtype=float) * false_image_values for n in range(n_false_examples)]
true_labels = [1 for n in range(n_true_examples)]
false_labels = [0 for n in range(n_false_examples)]


true_data = list(zip(true_images, true_labels))
false_data = list(zip(false_images, false_labels))
all_data = true_data + false_data
random.shuffle(all_data)

for tfrecord_loc in ['train.tfrecords', 'test.tfrecords']:
# for tfrecord_loc in ['temp.tfrecords']:

    if os.path.exists(tfrecord_loc):
        print('{} already exists - deleting'.format(tfrecord_loc))
        os.remove(tfrecord_loc)
    writer = tf.python_io.TFRecordWriter(tfrecord_loc)

    for example in all_data:
        image_to_tfrecord(matrix=example[0], label=example[1], writer=writer)
    writer.close()


"""
Computational graph
"""

batch = 32  # keeping it simple for now

train_features, train_labels = input(filename='train.tfrecords', mode='train', size=size, batch=batch, stratify=True, augment=False)
train_images = train_features['x']

test_features, test_labels = input(filename='test.tfrecords', mode='test', size=size, batch=batch, stratify=True, augment=False)
test_images = test_features['x']

"""
Execution
"""


with tf.train.MonitoredSession() as sess:

    train_images, train_labels = sess.run([train_images, train_labels])

    # print(train_images)
    print(train_labels)
    print('train images shape', train_images.shape)
    print('train labels shape', train_labels.shape)

    test_images, test_labels = sess.run([test_images, test_labels])

    print(test_labels)
    print('test images shape', test_images.shape)
    print('test labels shape', test_labels.shape)

# print(true_images)
# print(false_images)
# print(true_labels)
# print(false_labels)

# batch_images, batch_labels = sess.run([batch_images, batch_labels])

# print(batch_images)
# print(batch_labels)

#
# assert np.sum(strat_labels) == int(batch/2)
# assert np.mean(strat_images) == 0.0
#
# hopefully_true_images = strat_images[strat_labels == 1]
# assert np.min(hopefully_true_images) == true_image_values
