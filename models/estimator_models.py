import tensorflow as tf


def default_model_results(features, labels, mode, params):
    """

    Args:
        features ():
        labels ():
        mode ():
        params ():

    Returns:

    """
    input_layer = features["x"]
    tf.summary.image('augmented', input_layer, 1)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=params['conv1_filters'],
        kernel_size=params['conv1_kernel'],
        padding=params['padding'],
        activation=params['conv1_activation'],
        name='model/layer1/conv1')
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=params['pool1_size'],
        strides=params['pool1_strides'],
        name='model/layer1/pool1')

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=params['conv2_filters'],
        kernel_size=params['conv2_kernel'],
        padding=params['padding'],
        activation=params['conv2_activation'],
        name='model/layer2/conv2')
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=params['pool2_size'],
        strides=params['pool2_strides'],
        name='model/layer2/pool2')

    # Flatten tensor into a batch of vectors
    pool2_flat = tf.reshape(pool2, [-1, int(params['image_dim'] / 4) ** 2 * 64], name='model/layer2/flat')
    tf.summary.histogram('pool2_flat', pool2_flat)  # to visualise embedding of learned features

    # Dense Layer
    dense1 = tf.layers.dense(
        inputs=pool2_flat,
        units=params['dense1_units'],
        activation=params['dense1_activation'],
        name='model/layer3/dense1')

    # Add dropout operation
    dropout = tf.layers.dropout(
        inputs=dense1,
        rate=params['dense1_dropout'],
        training=mode == tf.estimator.ModeKeys.TRAIN)
    tf.summary.tensor_summary('dropout_summary', dropout)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=2, name='model/layer4/logits')
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('logits probabilities', tf.nn.softmax(logits))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Calculate Loss (for both TRAIN and EVAL modes)
    # required for EstimatorSpec
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

    return predictions, loss


def chollet_model_results(features, labels, mode, params):
    """

    Args:
        features ():
        labels ():
        mode ():
        params ():

    Returns:

    """
    input_layer = features["x"]
    tf.summary.image('augmented', input_layer, 1)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=params['conv1_filters'],
        kernel_size=params['conv1_kernel'],
        padding=params['padding'],
        activation=params['conv1_activation'],
        name='model/layer1/conv1')
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=params['pool1_size'],
        strides=params['pool1_strides'],
        name='model/layer1/pool1')

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=params['conv2_filters'],
        kernel_size=params['conv2_kernel'],
        padding=params['padding'],
        activation=params['conv2_activation'],
        name='model/layer2/conv2')
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=params['pool2_size'],
        strides=params['pool2_strides'],
        name='model/layer2/pool2')

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=params['conv3_filters'],
        kernel_size=params['conv3_kernel'],
        padding=params['padding'],
        activation=params['conv3_activation'],
        name='model/layer3/conv3')
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=params['pool3_size'],
        strides=params['pool3_strides'],
        name='model/layer3/pool3')

    # Flatten tensor into a batch of vectors
    pool3_flat = tf.reshape(pool3, [-1, int(params['image_dim'] / 8) ** 2 * 64], name='model/layer3/flat')

    # Dense Layer
    dense1 = tf.layers.dense(
        inputs=pool3_flat,
        units=params['dense1_units'],
        activation=params['dense1_activation'],
        name='model/layer4/dense1')

    # Add dropout operation
    dropout = tf.layers.dropout(
        inputs=dense1,
        rate=params['dense1_dropout'],
        training=mode == tf.estimator.ModeKeys.TRAIN)
    tf.summary.tensor_summary('dropout_summary', dropout)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=2, name='model/layer4/logits')
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('logits probabilities', tf.nn.softmax(logits))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Calculate Loss (for both TRAIN and EVAL modes)
    # required for EstimatorSpec
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

    return predictions, loss