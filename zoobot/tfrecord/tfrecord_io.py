import os
from functools import partial
import logging

import numpy as np
import tensorflow as tf

# from tfrecord.create_tfrecord import image_to_tfrecord


def load_dataset(example_loc, feature_spec, num_parallel_calls=1):
    # small wrapper around loading a TFRecord as a single tensor tuples
    logging.debug('tfrecord.io: Loading dataset from {}'.format(example_loc))
    dataset = tf.data.TFRecordDataset(example_loc)
    parse_function = partial(tf.parse_single_example, features=feature_spec)
    if isinstance(example_loc, str):
        return dataset.map(parse_function, num_parallel_calls=num_parallel_calls)  # Parse the record into tensors
    else:
        # see https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave 
         # read from all tfrecords in parallel, retrieving an even mix from all tfrecords, with 1 example from each record per cycle
         # note: mix is inversely proportional to length of each tfrecord, so they should be similar sizes!
        filenames_dataset = tf.data.Dataset.from_tensor_slices(example_loc)
        return filenames_dataset.interleave(
            lambda file_loc: tf.data.TFRecordDataset(file_loc).map(parse_function), 
            cycle_length=len(example_loc), 
            block_length=1, 
            num_parallel_calls=num_parallel_calls)
        

# TODO convert this to a proper test of dataset readability?
# if __name__ == '__main__':
#
#     example_image_data = np.array(np.ones((50, 50, 3), dtype=float))
#     tfrecord_dir = '/data/repos/zoobot/zoobot/get_catalogs/tfrecord'
#     label = 1
#     save_loc = '{}/example.tfrecords'.format(tfrecord_dir)
#     if os.path.exists(save_loc):
#         os.remove(save_loc)
#     assert not os.path.exists(save_loc)
#     writer = tf.python_io.TFRecordWriter(save_loc)
#     image_to_tfrecord(example_image_data, label, writer)
#     assert os.path.exists(save_loc)
#     writer.close()  # very important - will give 'DataLoss' error if writer not closed
#
#     feature_spec = matrix_label_feature_spec(size=50)
#     example_features = read_first_example(save_loc, feature_spec)
#     label = example_features['label']
#     image = example_features['matrix']
#
#     sess = tf.Session()
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     tf.train.start_queue_runners(sess=sess)
#     label, image = sess.run([label, image])
#
#     print(label, image)
