import logging
import sys

import tensorflow as tf

from zoobot.estimators import losses


class BayesianModel(tf.keras.Model):

    def __init__(
            self,
            image_dim,
            output_dim,
            schema,
            conv1_filters=32,
            conv1_kernel=1,
            conv1_activation=tf.nn.relu,
            conv2_filters=32,
            conv2_kernel=3,
            conv2_activation=tf.nn.relu,
            conv3_filters=16,
            conv3_kernel=3,
            conv3_activation=tf.nn.relu,
            dense1_units=128,
            dense1_dropout=0.5,
            dense1_activation=tf.nn.relu,
            predict_dropout=0.5,
            regression=False,
            log_freq=10,

    ):
        super(BayesianModel, self).__init__()

        self.output_dim = output_dim  # n final neuron
        self.image_dim = image_dim
        self.conv3_filters = conv3_filters  # actually useful elsewhere for the reshape
        self.log_freq = log_freq
        # self.logging_hooks = logging_hooks(self)  # TODO strange error with passing this to estimator in params
        self.logging_hooks = [None, None, None]
 
        dropout_rate = 0  # no dropout on conv layers
        regularizer = None
        padding = 'same'
        pool1_size = 2
        pool1_strides = 2
        pool2_size = 2
        pool2_strides = 2
        pool3_size = 2
        pool3_strides = 2

        self.conv1 = tf.keras.layers.Convolution2D(
            filters=conv1_filters,
            kernel_size=[conv1_kernel, conv1_kernel],
            padding=padding,
            activation=conv1_activation,
            kernel_regularizer=regularizer,
            name='model/layer1/conv1')
        self.drop1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.conv1b = tf.keras.layers.Convolution2D(
            filters=conv1_filters,
            kernel_size=[conv1_kernel, conv1_kernel],
            padding=padding,
            activation=conv1_activation,
            kernel_regularizer=regularizer,
            name='model/layer1/conv1b')
        self.drop1b = tf.keras.layers.Dropout(rate=dropout_rate)
        self.pool1 = tf.keras.layers.MaxPooling2D(
            pool_size=[pool1_size, pool1_size],
            strides=pool1_strides,
            name='model/layer1/pool1')

        self.conv2 = tf.keras.layers.Convolution2D(
            filters=conv2_filters,
            kernel_size=[conv2_kernel, conv2_kernel],
            padding=padding,
            activation=conv2_activation,
            kernel_regularizer=regularizer,
            name='model/layer2/conv2')
        self.drop2 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.conv2b = tf.keras.layers.Convolution2D(
            filters=conv2_filters,
            kernel_size=[conv2_kernel, conv2_kernel],
            padding=padding,
            activation=conv2_activation,
            kernel_regularizer=regularizer,
            name='model/layer2/conv2b')
        self.drop2b = tf.keras.layers.Dropout(rate=dropout_rate)
        self.pool2 = tf.keras.layers.MaxPooling2D(
            pool_size=pool2_size,
            strides=pool2_strides,
            name='model/layer2/pool2')

        self.conv3 = tf.keras.layers.Convolution2D(
            filters=conv3_filters,
            kernel_size=[conv3_kernel, conv3_kernel],
            padding=padding,
            activation=conv3_activation,
            kernel_regularizer=regularizer,
            name='model/layer3/conv3')
        self.drop3 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.pool3 = tf.keras.layers.MaxPooling2D(
            pool_size=[pool3_size, pool3_size],
            strides=pool3_strides,
            name='model/layer3/pool3')

        # identical to conv3
        self.conv4 = tf.keras.layers.Convolution2D(
            filters=conv3_filters,
            kernel_size=[conv3_kernel, conv3_kernel],
            padding=padding,
            activation=conv3_activation,
            kernel_regularizer=regularizer,
            name='model/layer4/conv4')
        self.drop4 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.pool4 = tf.keras.layers.MaxPooling2D(
            pool_size=[pool3_size, pool3_size],
            strides=pool3_strides,
            name='model/layer4/pool4')

        self.dense1 = tf.keras.layers.Dense(
            units=dense1_units,
            activation=dense1_activation,
            kernel_regularizer=regularizer,
            name='model/layer4/dense1')
        self.dropout_final = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dense_final = tf.keras.layers.Dense(
            units=self.output_dim,  # num outputs
            name='model/layer5/dense1')

    # really important to use call, not __call__, or the dataset won't be made in graph mode and you'll get eager/symbolic mismatch error
    @tf.function
    def call(self, x, training=None):

        dropout_on = True  # dropout always on, regardless of training arg (required by keras)
        x = self.conv1(x)
        x = self.drop1(x, training=dropout_on)
        x = self.conv1b(x)
        x = self.drop1b(x, training=dropout_on)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.drop2(x, training=dropout_on)
        x = self.conv2b(x)
        x = self.drop2b(x, training=dropout_on)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.drop3(x, training=dropout_on)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.drop4(x, training=dropout_on)
        x = self.pool4(x)

        """
        Flatten tensor into a batch of vectors
        Start with image_dim shape. NB, does not change with channels: just alters num of first filters.
        2 * 2 * 2 = 8 factor reduction in shape from pooling, assuming stride 2 and pool_size 2
        length ^ 2 to make shape 1D
        64 filters in final layer
        """
        x = tf.reshape(x, [-1, int(self.image_dim / 16) ** 2 * self.conv3_filters], name='model/layer4/flat')

        x = self.dropout_final(x, training=dropout_on)
        x = self.dense_final(x)

        # normalise predictions by question (TODO hardcoded!)
        x = tf.concat([ tf.nn.softmax(x[:, :2]), tf.nn.softmax(x[:, 2:]) ], axis=1)
        # tf.summary.histogram('normalised_smooth_prediction', x[:, 0])
        # tf.summary.histogram('normalised_spiral_prediction', x[:, 2])

        return x


    # TODO move to shared utilities
    # TODO duplicated with input_utils
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])

@tf.function
def squared_error(labels, predictions):
    # already filtered to have only the appropriate columns
    # print(tf.shape(labels, name='labels_shape'))
    # print(tf.shape(predictions, name='predictions_shape'))
    # assert tf.shape(labels)[1] == tf.shape(predictions)[1]

    # spiral_observed_fracs = labels[:, 2:]/tf.expand_dims(tf.reduce_sum(labels[:, 2:], axis=1), axis=1)
    # smooth_total = tf.reduce_sum(input_tensor=labels[:, :2], axis=1)
    # print(labels[:, 2:])
    total = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    # tf.summary.histogram('smooth_total', smooth_total)
    # tf.summary.histogram('total', total)
    # smooth_observed_fracs = labels[:, 0]/smooth_total


    observed_fracs = tf.math.divide_no_nan(labels, total)  # WARNING need to check into this
    # observed_vote_fractions = tf.concat([ labels[:, :2]/tf.expand_dims(tf.reduce_sum(labels[:, :2], axis=1), axis=1), labels[:, 2:]/tf.expand_dims(tf.reduce_sum(labels[:, 2:], axis=1), axis=1) ], axis=1)
    # tf.summary.histogram('smooth_observed_fracs', smooth_observed_fracs)
    # tf.summary.histogram('observed_fracs', observed_fracs)

    # squared_smooth_error = (smooth_observed_fracs - predictions[:, 0]) ** 2
    sq_error = (observed_fracs - predictions) ** 2

    # tf.summary.histogram('squared_smooth_error', squared_smooth_error)
    # tf.summary.histogram('squared_spiral_error', squared_spiral_error)

    # tf.summary.scalar('squared_smooth_mse', tf.reduce_mean(squared_smooth_error))
    # tf.summary.scalar('squared_spiral_mse', tf.reduce_mean(squared_spiral_error))

    return sq_error


class CustomMSEByColumn(tf.keras.metrics.Metric):

    def __init__(self, name, start_col, end_col, **kwargs):
        print(f'Name: {name}, start {start_col}, end {end_col}')
        super(CustomMSEByColumn, self).__init__(name=name, **kwargs)
        self.mse = self.add_weight(name=name, initializer='zeros')
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        self.total_se = tf.Variable(0., dtype=tf.float32)
        self.start_col = start_col
        self.end_col = end_col

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_se.assign_add(tf.reduce_sum(squared_error(y_true[:, self.start_col:self.end_col+1], y_pred[:, self.start_col:self.end_col+1])))
        self.batch_count.assign_add(tf.shape(y_true)[0])
        self.mse.assign(self.total_se/tf.cast(self.batch_count, dtype=tf.float32))

    def result(self):
        return self.mse

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.mse.assign(0.)
        self.batch_count.assign(0)
        self.total_se.assign(0.)  


# class CustomSmoothMSE(tf.keras.metrics.Metric):

#     def __init__(self, name='custom_smooth_MSE', **kwargs):
#         super(CustomSmoothMSE, self).__init__(name=name, **kwargs)
#         self.mse = self.add_weight(name='smooth_mse', initializer='zeros')
#         self.batch_count = tf.Variable(0, dtype=tf.int32)
#         self.total_se = tf.Variable(0., dtype=tf.float32)

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         self.total_se.assign_add(tf.reduce_sum(squared_smooth_error(y_true, y_pred)))
#         self.batch_count.assign_add(tf.shape(y_true)[0])
#         self.mse.assign(self.total_se/tf.cast(self.batch_count, dtype=tf.float32))

#     def result(self):
#         return self.mse

#     def reset_states(self):
#       # The state of the metric will be reset at the start of each epoch.
#         self.mse.assign(0.)
#         self.batch_count.assign(0)
#         self.total_se.assign(0.)

# class CustomSpiralMSE(tf.keras.metrics.Metric):

#     def __init__(self, name='custom_spiral_MSE', **kwargs):
#         super(CustomSpiralMSE, self).__init__(name=name, **kwargs)
#         self.mse = self.add_weight(name='spiral_mse', initializer='zeros')
#         self.batch_count = tf.Variable(0, dtype=tf.int32)
#         self.total_se = tf.Variable(0., dtype=tf.float32)

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         self.total_se.assign_add(tf.reduce_sum(squared_spiral_error(y_true, y_pred)))
#         self.batch_count.assign_add(tf.shape(y_true)[0])
#         self.mse.assign(self.total_se/tf.cast(self.batch_count, dtype=tf.float32))

#     def result(self):
#         return self.mse

#     def reset_states(self):
#       # The state of the metric will be reset at the start of each epoch.
#         self.mse.assign(0.)
#         self.batch_count.assign(0)
#         self.total_se.assign(0.)

    # train_accuracy = tf.keras.metrics.MeanSquaredError('train_mse')
    # test_accuracy = tf.keras.metrics.MeanSquaredError('test_mse')
    # 'smooth_observed_fracs_eval': tf.metrics.root_mean_squared_error(),
    # 'spiral_observed_fracs_eval': tf.metrics.root_mean_squared_error()
        
# def get_gz_binomial_eval_metric_ops(self, labels, predictions):
    # raise NotImplementedError('Needs to be updated for multi-label! Likely to replace in TF2.0')
    # will probably be callable/subclass rather than implemented here 
    # record distribution of predictions for tensorboard
    # tf.summary.histogram('yes_votes', labels[0, :])
    # tf.summary.histogram('total_votes', labels[1, :])
    # assert labels.dtype == tf.int64
    # assert predictions['prediction'].dtype == tf.float32
    # observed_vote_fraction = tf.cast(labels[:, 0], dtype=tf.float32) / tf.cast(labels[:, 1], dtype=tf.float32)
    # tf.summary.histogram('observed vote fraction', observed_vote_fraction)
    # return {"rmse": tf.metrics.root_mean_squared_error(observed_vote_fraction, predictions['prediction'])}

# def logging_hooks(model_config):
#     train_tensors = {
#         'labels': 'labels',
#         # 'logits': 'logits',  may not always exist? TODO
#         "probabilities": 'softmax',
#         'mean_loss': 'mean_loss'
#     }
#     train_hook = tf.train.LoggingTensorHook(
#         tensors=train_tensors, every_n_iter=model_config.log_freq)

#     # eval_hook = train_hook
#     eval_hook = tf.train.LoggingTensorHook(
#         tensors=train_tensors, every_n_iter=model_config.log_freq)

#     prediction_tensors = {}
#     # [prediction_tensors.update({'sample_{}/predictions'.format(n): 'sample_{}/predictions'.format(n)}) for n in range(3)]

#     prediction_hook = tf.train.LoggingTensorHook(
#         tensors=prediction_tensors, every_n_iter=model_config.log_freq
#     )

#     return [train_hook], [eval_hook], [prediction_hook]  # estimator expects lists of logging hooks
