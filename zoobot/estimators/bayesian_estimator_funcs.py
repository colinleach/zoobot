
import tensorflow as tf
from tensorboard import summary as tensorboard_summary


def four_layer_binary_classifier(features, labels, mode, params):
    """
    Estimator wrapper function for four-layer cnn performing binary classification
    Details (e.g. neurons, activation funcs, etc) controlled by 'params'

    Args:
        features ():
        labels ():
        mode ():
        params ():

    Returns:

    """

    predictions, loss = bayesian_cnn(features, labels, mode, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params['optimizer'](learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    tensorboard_summary.pr_curve_streaming_op(
        name='spirals',
        labels=labels,
        predictions=predictions['probabilities'][:, 1],
    )
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = get_eval_metric_ops(labels, predictions)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def input_to_dense(features, params):
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

    return dense1


def dense_to_prediction(dense1, labels, params, dropout_on, name=True):

    # Add dropout operation
    dropout = tf.layers.dropout(
        inputs=dense1,
        rate=params['dense1_dropout'],
        training=dropout_on)
    tf.summary.tensor_summary('dropout_summary', dropout)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=2, name='logits')

    tf.summary.histogram('logits', logits)
    tf.summary.histogram('logits probabilities', tf.nn.softmax(logits))

    prediction = {
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        "predictions": tf.nn.softmax(logits)[:, 0],  # keep only one softmax per subject. Softmax sums to 1 in TF.
        # TODO PR the docs about this?
    }
    if labels is not None:
        prediction.update({
            'labels': tf.identity(labels, name='labels'),  #  these are None in predict mode
            "classes": tf.argmax(input=logits, axis=1, name='classes'),
        })

    return logits, prediction


def bayesian_cnn(features, labels, mode, params):
    """
    Model function of four-layer CNN
    Can be used in isolation or called within an estimator e.g. four_layer_binary_classifier

    Args:
        features ():
        labels ():
        mode ():
        params ():

    Returns:

    """

    dense1 = input_to_dense(features, params)

    # here, the model branches/loops at test time
    if mode == tf.estimator.ModeKeys.PREDICT:

        prediction_tensors = []

        def new_predictions():
            _, predictions = dense_to_prediction(dense1, labels, params, dropout_on=True)
            prediction_tensors.append(predictions)

        n = 0
        tf.while_loop(tf.less(n, 4), new_predictions())

        # _, predictions_a = dense_to_prediction(dense1, labels, params, dropout_on=True)  # always apply dropout. Rename tensors to avoid conflicts.
        # _, predictions_b = dense_to_prediction(dense1, labels, params, dropout_on=True)
        # _, predictions_c = dense_to_prediction(dense1, labels, params, dropout_on=True)
        #
        # prediction_tensors.append(predictions_a['predictions'])  # TODO fix bad naming
        # prediction_tensors.append(predictions_b['predictions'])
        # prediction_tensors.append(predictions_c['predictions'])

        all_prediction_tensors = tf.concat(prediction_tensors, axis=0, name='all_predictions')  # dimensions [batch_size, n_samples]
        predictions.update({'all_predictions': all_prediction_tensors})  # use labels/classes/probs from last sample
        return predictions, None  # no loss needed


    # Calculate Loss (for both TRAIN and EVAL modes)
    # required for EstimatorSpec
    else:
        logits, predictions = dense_to_prediction(dense1, labels, params,dropout_on=mode == tf.estimator.ModeKeys.TRAIN)

        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits, name='model/layer4/loss')
        mean_loss = tf.reduce_mean(loss, name='mean_loss')
        return predictions, mean_loss


def get_eval_metric_ops(labels, predictions):

    # record distribution of predictions for tensorboard
    tf.summary.histogram('Probabilities', predictions['probabilities'])
    # tf.summary.histogram('Classes', predictions['classes'])

    return {
        "acc/accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]),
        "pr/auc": tf.metrics.auc(labels=labels, predictions=predictions['classes']),
        "acc/mean_per_class_accuracy": tf.metrics.mean_per_class_accuracy(labels=labels, predictions=predictions['classes'], num_classes=2),
        'pr/precision': tf.metrics.precision(labels=labels, predictions=predictions['classes']),
        'pr/recall': tf.metrics.recall(labels=labels, predictions=predictions['classes']),
        'confusion/true_positives': tf.metrics.true_positives(labels=labels, predictions=predictions['classes']),
        'confusion/true_negatives': tf.metrics.true_negatives(labels=labels, predictions=predictions['classes']),
        'confusion/false_positives': tf.metrics.false_positives(labels=labels, predictions=predictions['classes']),
        'confusion/false_negatives': tf.metrics.false_negatives(labels=labels, predictions=predictions['classes'])
    }
