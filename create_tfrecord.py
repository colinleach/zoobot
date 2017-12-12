import tensorflow as tf
import numpy as np


# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#
# def array_to_tfrecord(image, filename):
#     writer = tf.python_io.TFRecordWriter(filename)
#
#     image_raw = image.tostring()
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'height': 3,
#         'width': 3,
#         'depth': 3,
#         'label': 1,
#         'image_raw': _bytes_feature(image_raw)}))
#     writer.write(example.SerializeToString())
#     writer.close()
#
#
# def main(unused_args):
#     example_image_data = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])
#     array_to_tfrecord(example_image_data, 'example.tfrecords')
#
# tf.app.run(main=main)

# load up some dataset. Could be anything but skdata is convenient.

from tqdm import tqdm
import numpy as np
import tensorflow as tf

image = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])
# image = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype='int64')
flat_image = np.reshape(image, image.size)
print(flat_image.shape)
flat_image_list = list(flat_image)

all_vectors = [flat_image_list for x in range(5)]
all_labels = np.ones(5, dtype=int)

writer = tf.python_io.TFRecordWriter("floats.tfrecords")
# iterate over each example
# wrap with tqdm for a progress bar
for example_idx in tqdm(range(5)):
    print(example_idx)
    features = all_vectors[example_idx]
    label = all_labels[example_idx]

    # construct the Example proto boject
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
          # Features contains a map of string to Feature proto objects
          feature={
            # A Feature contains one of either a int64_list,
            # float_list, or bytes_list
            'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label])),
            'image': tf.train.Feature(
                float_list=tf.train.FloatList(value=features)),
    }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)
