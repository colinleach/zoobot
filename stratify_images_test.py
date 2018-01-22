import tensorflow as tf
import numpy as np

from input_utils import stratify_images, stratify_images_auto, rejection_sample

import matplotlib.pyplot as plt

"""
Computational graph
"""


batch_size = 32  # keeping it simple for now

all_data = tf.constant([1. for n in range(800)] + [0. for n in range(200)])
all_labels = tf.constant([1 for n in range(800)] + [0 for n in range(200)])
dataset = tf.data.Dataset.from_tensor_slices(
    {'data': all_data,
     'labels': all_labels}
)

dataset = dataset.shuffle(buffer_size=1000)
# dataset = dataset.batch(32)
dataset = dataset.repeat(1)
batch = dataset.make_one_shot_iterator().get_next()

images = batch['data']
labels = batch['labels']

stratified_result = stratify_images_auto(images, labels, batch_size)


"""
Execution
"""

with tf.train.MonitoredSession() as sess:

    all_labels = []
    all_means = []
    while not sess.should_stop():
        result = sess.run(stratified_result)
        data_batch = result[0]
        label_batch = result[1]
        print(label_batch)
        all_labels.append(label_batch)
        all_means.append(np.array(all_labels).mean())

plt.plot(all_means)
plt.show()
