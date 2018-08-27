import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf


def load_examples_from_tfrecord(tfrecord_locs, feature_spec, n_examples=None):
    # see http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
    with tf.Session() as sess:
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(tfrecord_locs, num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        example = tf.parse_single_example(serialized_example, features=feature_spec)

         # initialize all global and local variables
        # this op must be defined (not only executed) within the session
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        logging.debug('Starting queue to load examples from {}'.format(tfrecord_locs))
        threads = tf.train.start_queue_runners(coord=coord, daemon=False)  # block exit until closed
        logging.debug(threads)
        logging.debug('Queue started')

        # execute
        if n_examples is None:  # load full record
            data = []
            while True:
    
                try:
                    loaded_example = sess.run(example)
                    data.append(loaded_example)
                except tf.errors.OutOfRangeError:
                    logging.debug('tfrecords {} exhausted'.format(tfrecord_locs))
                    break
        else:
            logging.debug('Loading the first {} examples from {}'.format(n_examples, tfrecord_locs))
            data = [sess.run(example) for n in range(n_examples)]

        # weird behaviour when used with pytest - threads may not stop on test end. Be explicit.
        coord.request_stop() # ask the threads to stop
        coord.join(threads)  # wait for them to actually do it

    return data


def matrix_label_feature_spec(size, channels):
    return {
        "matrix": tf.FixedLenFeature((size * size * channels), tf.float32),
        "label": tf.FixedLenFeature((), tf.int64)
        }


def matrix_id_feature_spec(size, channels):
    return {
        "matrix": tf.FixedLenFeature((size * size * channels), tf.float32),
        "id_str": tf.FixedLenFeature((), tf.string)
        }


def matrix_label_id_feature_spec(size, channels):
    return {
        "matrix": tf.FixedLenFeature((size * size * channels), tf.float32),
        "label": tf.FixedLenFeature((), tf.int64),
        "id_str": tf.FixedLenFeature((), tf.string)
        }


# not required, use tf.parse_single_example directly
# def parse_example(example, size, channels):
#     features = {
#         'matrix': tf.FixedLenFeature((size * size * channels), tf.float32),
#         'label': tf.FixedLenFeature([], tf.int64),
#         }

#     return tf.parse_single_example(example, features=features)


# these are actually not related to reading a tfrecord, they are very general
def show_examples(examples, size, channels):
    # simple wrapper for pretty example plotting
    # TODO make plots in a grid rather than vertical column
    fig, axes = plt.subplots(nrows=len(examples), figsize=(4, len(examples) * 3))
    for n, example in enumerate(examples):
        show_example(example, size, channels, ax=axes[n])
    fig.tight_layout()
    return fig, axes


def show_example(example, size, channels, ax):  # modifies ax inplace
    # saved as floats but truly int, show as int
    im = example['matrix'].reshape(size, size, channels) 
    label = example['label']
    name_mapping = {
        0: 'Feat.',
        1: 'Smooth'
    }
    ax.imshow(im)
    ax.text(60, 110, name_mapping[label], fontsize=16, color='r')
