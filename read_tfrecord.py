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
            'matrix': tf.FixedLenFeature([64 ** 2 * 3], tf.float32),
            't04_spiral_a08_spiral_weighted_fraction': tf.FixedLenFeature([], tf.float32)
        })
    # now return the converted data
    label = features['label']
    image = features['matrix']
    spiral_fraction = features['t04_spiral_a08_spiral_weighted_fraction']

    return label, image, spiral_fraction

# returns symbolic label and matrix
example_loc = '/data/galaxy_zoo/gz2/tfrecord/spiral_64.tfrecord'
label, image, spiral_fraction = read_and_decode_single_example(example_loc)

sess = tf.Session()

# Required. See below for explanation
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

# # grab examples back.
# # first example from file
# label_val_1, image_val_1 = sess.run([label, image])
# # second example from file
# label_val_2, image_val_2 = sess.run([label, image])
#
# print(label_val_1, label_val_2)

# https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

# groups examples into batches randomly
# https://www.tensorflow.org/api_docs/python/tf/train/shuffle_batch
images_batch, labels_batch, spiral_fraction_batch = tf.train.shuffle_batch(
    [image, label, spiral_fraction], batch_size=128,
    capacity=2000,
    min_after_dequeue=1000)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
labels, images, spiral_fractions = sess.run([labels_batch, images_batch, spiral_fraction_batch])

square_images = images.reshape([-1, 64, 64])
print(labels)
print(spiral_fractions)
print(square_images[0])