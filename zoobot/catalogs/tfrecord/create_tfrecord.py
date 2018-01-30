
from tqdm import tqdm
import numpy as np
import tensorflow as tf


def image_to_tfrecord(matrix, label, writer, extra_data=None):
    """
    Save an image, label and any additional data to TFRecord. If no record exists, create one.

    Args:
        matrix (np.array): pixel data. Floats in shape [height, width, depth]
        label (int): class of image
        writer ():
        extra_data (dict): of form {'name of feature to save': value(s) to be saved. Int, float, or np.array only

    Returns:
        None
    """

    flat_matrix = np.reshape(matrix, matrix.size)

    # A Feature contains one of either a int64_list,
    # float_list, or bytes_list
    # Each of these will be preserved in the TFRecord
    label_feature = int_to_feature(label)

    matrix_feature = float_list_to_feature(flat_matrix)

    # Expects TensorFlow data format convention, "Height-Width-Depth".
    if matrix.shape[2] > matrix.shape[0]:
        raise Exception('Fatal error: image not in height-width-depth convention')

    height_feature = int_to_feature(matrix.shape[0])
    width_feature = int_to_feature(matrix.shape[1])
    channels_feature = int_to_feature(matrix.shape[2])

    features_to_save = {
        'label': label_feature,
        'matrix': matrix_feature,
        'channels': channels_feature,
        'height': height_feature,
        'width': width_feature
    }

    if extra_data:
        extra_data = extra_data.copy()  # avoid mutating input dict
        for name, value in extra_data.items():
            extra_data[name] = value_to_feature(value)
        features_to_save.update(extra_data)

    # construct the Example proto boject
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
            feature=features_to_save))
    # use the proto object to serialize the example to a string

    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)


def value_to_feature(value):
    """
    Helper function to convert value of unknown type into proto feature

    Args:
        value (): value to be converted

    Returns:
        (tf.train.Feature) encoding of value, according to value type.
    """
    if type(value) == int:
        return int_to_feature(value)
    elif type(value) == float:
        return float_to_feature(value)
    elif type(value) == list or type(value) == np.ndarray:
        return float_list_to_feature(value)
    else:
        raise Exception('Fatal error: {} feature type not understood'.format(value))


def int_to_feature(int_to_save):
    return tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int_to_save]))


def float_to_feature(float_to_save):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[float_to_save]))


def float_list_to_feature(floats_to_save):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=floats_to_save))
