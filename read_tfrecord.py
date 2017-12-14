import tensorflow as tf


def read_and_decode_single_example(filename):
    # first construct a queue containing a list of filenames.
    # this lets a user split up the dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([], tf.int64),
            'matrix': tf.FixedLenFeature([9], tf.float32)
        })
    # now return the converted data
    label = features['label']
    image = features['matrix']
    return label, image

# returns symbolic label and matrix
label, image = read_and_decode_single_example("floats.tfrecords")

sess = tf.Session()

# Required. See below for explanation
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

# grab examples back.
# first example from file
# label_val_1, image_val_1 = sess.run([label, matrix])
# # second example from file
# label_val_2, image_val_2 = sess.run([label, matrix])
#
# image_1 = image_val_1.reshape([3, 3])
# image_2 = image_val_2.reshape([3, 3])
#
# print(label_val_1, image_1)
# print(label_val_2, image_2)

# https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

# groups examples into batches randomly
# https://www.tensorflow.org/api_docs/python/tf/train/shuffle_batch
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=128,
    capacity=2000,
    min_after_dequeue=1000)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
labels, images = sess.run([labels_batch, images_batch])

square_images = images.reshape([-1, 3, 3])
print(square_images)
