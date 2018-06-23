import tensorflow as tf
from tensorflow.python.saved_model import signature_constants


def estimator_wrapper(features, labels, mode, params):
    # estimator model funcs are only allowed to have (features, labels, params) arguments
    # re-order the arguments internally to allow for the custom class to be passed around
    # params is really the model class
    return params.entry_point(features, labels, mode)  # must have exactly the args (features, labels)


class BayesianBinaryModel():

    def __init__(
            self,
            image_dim,
            learning_rate=0.001,
            optimizer=tf.train.AdamOptimizer,
            conv1_filters=32,
            conv1_kernel=1,
            conv2_filters=32,
            conv2_kernel=3,
            conv3_filters=16,
            conv3_kernel=3,
            dense1_units=128,
            dense1_dropout=0.5,
            log_freq=10,
    ):
        self.image_dim = image_dim
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.conv1_filters = conv1_filters
        self.conv1_kernel = conv1_kernel
        self.conv2_filters = conv2_filters
        self.conv2_kernel = conv2_kernel
        self.conv3_filters = conv3_filters
        self.conv3_kernel = conv3_kernel
        self.dense1_units = dense1_units
        self.dense1_dropout = dense1_dropout
        self.conv1_activation=tf.nn.relu
        self.conv2_activation=tf.nn.relu
        self.conv3_activation=tf.nn.relu
        self.dense1_activation=tf.nn.relu
        self.pool1_size=2
        self.pool1_strides=2
        self.pool2_size=2
        self.pool2_strides=2
        self.pool3_size=2
        self.pool3_strides=2
        self.padding = 'same'
        self.log_freq = log_freq
        self.model_fn = self.four_layer_binary_classifier
        # self.logging_hooks = logging_hooks(self)  # TODO strange error with passing this to estimator in params
        self.logging_hooks = [None, None, None]
        self.entry_point = self.four_layer_binary_classifier


    def four_layer_binary_classifier(self, features, labels, mode):
        """
        Estimator wrapper function for four-layer cnn performing binary classification
        Details (e.g. neurons, activation funcs, etc) controlled by 'params'

        Args:
            features ():
            labels ():
            mode ():

        Returns:

        """
        response, loss = self.bayesian_cnn(features, labels, mode)

        if mode == tf.estimator.ModeKeys.PREDICT:
            with tf.variable_scope('predict'):
                export_outputs = {
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({
                        'predictions_for_true': response['predictions_for_true']
                        })
                }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=response, export_outputs=export_outputs)

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope('train'):
                optimizer = self.optimizer(learning_rate=self.learning_rate)  # TODO adaptive learning rate
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        else:  # must be EVAL mode
            with tf.variable_scope('eval'):

                # Add evaluation metrics (for EVAL mode)
                eval_metric_ops = get_eval_metric_ops(labels, response)
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    def bayesian_cnn(self, features, labels, mode):
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

        dense1 = input_to_dense(features, self)  # run from input to dense1 output

        # if predict mode, feedforward from dense1 SEVERAL TIMES. Save all predictions under 'all_predictions'.
        if mode == tf.estimator.ModeKeys.PREDICT:

            with tf.variable_scope("sample"):
                # Feedforward from dense1. Always apply dropout.
                _, sample_predictions = dense_to_prediction(dense1, labels, self, dropout_on=True)
            return sample_predictions, None  # no loss, as labels not known (in general)

        else:  # Calculate Loss for TRAIN and EVAL modes)
            # only feedforward once for one set of predictions
            logits, response = dense_to_prediction(dense1, labels, self, dropout_on=mode == tf.estimator.ModeKeys.TRAIN)
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
            onehot_labels = tf.stop_gradient(onehot_labels)  # don't find the gradient of the labels (e.g. adversarial)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits, name='model/layer4/loss')
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits, name='model/layer4/loss')
            mean_loss = tf.reduce_mean(loss, name='mean_loss')

            # create dummy variables that match names in predict mode
            # TODO this is potentially wasteful as we don't actually need the feedforwards. Unclear if it executes - check.
            with tf.variable_scope("sample"):
                _, _ = dense_to_prediction(dense1, labels, self, dropout_on=True)

            return response, mean_loss


def input_to_dense(features, model):
    """

    Args:
        features ():
        model (BayesianBinaryModel):

    Returns:

    """
    input_layer = features["x"]
    tf.summary.image('model_input', input_layer, 1)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=model.conv1_filters,
        kernel_size=[model.conv1_kernel, model.conv1_kernel],
        padding=model.padding,
        activation=model.conv1_activation,
        name='model/layer1/conv1')
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[model.pool1_size, model.pool1_size],
        strides=model.pool1_strides,
        name='model/layer1/pool1')

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=model.conv2_filters,
        kernel_size=[model.conv2_kernel, model.conv2_kernel],
        padding=model.padding,
        activation=model.conv2_activation,
        name='model/layer2/conv2')
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=model.pool2_size,
        strides=model.pool2_strides,
        name='model/layer2/pool2')

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=model.conv3_filters,
        kernel_size=[model.conv3_kernel, model.conv3_kernel],
        padding=model.padding,
        activation=model.conv3_activation,
        name='model/layer3/conv3')
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[model.pool3_size, model.pool3_size],
        strides=model.pool3_strides,
        name='model/layer3/pool3')

    """
    Flatten tensor into a batch of vectors
    Start with image_dim shape, 1 channel
    2 * 2 * 2 = 8 factor reduction in shape from pooling, assuming stride 2 and pool_size 2
    length ^ 2 to make shape 1D
    64 filters in final layer
    """
    pool3_flat = tf.reshape(pool3, [-1, int(model.image_dim / 8) ** 2 * model.conv3_filters], name='model/layer3/flat')

    # Dense Layer
    dense1 = tf.layers.dense(
        inputs=pool3_flat,
        units=model.dense1_units,
        activation=model.dense1_activation,
        name='model/layer4/dense1')

    return dense1


def dense_to_prediction(dense1, labels, params, dropout_on):

    # Add dropout operation
    dropout = tf.layers.dropout(
        inputs=dense1,
        rate=params.dense1_dropout,
        training=dropout_on)
    tf.summary.tensor_summary('dropout_summary', dropout)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=2, name='logits')
    tf.summary.histogram('logits_false', logits[:, 0])  # first dimension is batch, second is class (0 = featured here)
    tf.summary.histogram('logits_true', logits[:, 1])

    softmax = tf.nn.softmax(logits, name='softmax')
    softmax_true_score = softmax[:, 1]  # predicts if label = 1 i.e. smooth
    tf.summary.histogram('softmax_true_score', softmax_true_score)

    response = {
        "probabilities": softmax,
        "predictions_for_true": softmax_true_score,  # keep only one softmax per subject. Softmax sums to 1
    }
    if labels is not None:
        response.update({
            'labels': tf.identity(labels, name='labels'),  #  these are None in predict mode
            "classes": tf.argmax(input=logits, axis=1, name='classes'),
        })

    return logits, response


def get_eval_metric_ops(labels, predictions):
    # record distribution of predictions for tensorboard
    tf.summary.histogram('Probabilities', predictions['probabilities'])
    tf.summary.histogram('Classes', predictions['classes'])

    # validation loss is added behind-the-scenes by TensorFlow
    return {
        "acc/accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]),
        "acc/mean_per_class_accuracy": tf.metrics.mean_per_class_accuracy(labels=labels,
                                                                          predictions=predictions['classes'],
                                                                          num_classes=2),
        "pr/auc": tf.metrics.auc(labels=labels, predictions=predictions['classes']),
        'pr/precision': tf.metrics.precision(labels=labels, predictions=predictions['classes']),
        'pr/recall': tf.metrics.recall(labels=labels, predictions=predictions['classes']),
        'confusion/true_positives': tf.metrics.true_positives(labels=labels, predictions=predictions['classes']),
        'confusion/true_negatives': tf.metrics.true_negatives(labels=labels, predictions=predictions['classes']),
        'confusion/false_positives': tf.metrics.false_positives(labels=labels, predictions=predictions['classes']),
        'confusion/false_negatives': tf.metrics.false_negatives(labels=labels, predictions=predictions['classes']),
        'sanity/predictions_below_5%': tf.metrics.percentage_below(
            values=predictions['probabilities'][:, 0],
            threshold=0.05),
        'sanity/predictions_above_95%': tf.metrics.percentage_below(
            values=1 - predictions['probabilities'][:, 0],
            threshold=0.05)
    }


def logging_hooks(model_config):
    train_tensors = {
        'labels': 'labels',
        # 'logits': 'logits',  may not always exist? TODO
        "probabilities": 'softmax',
        'mean_loss': 'mean_loss'
    }
    train_hook = tf.train.LoggingTensorHook(
        tensors=train_tensors, every_n_iter=model_config.log_freq)

    # eval_hook = train_hook
    eval_hook = tf.train.LoggingTensorHook(
        tensors=train_tensors, every_n_iter=model_config.log_freq)

    prediction_tensors = {}
    # [prediction_tensors.update({'sample_{}/predictions'.format(n): 'sample_{}/predictions'.format(n)}) for n in range(3)]

    prediction_hook = tf.train.LoggingTensorHook(
        tensors=prediction_tensors, every_n_iter=model_config.log_freq
    )

    return [train_hook], [eval_hook], [prediction_hook]  # estimator expects lists of logging hooks
