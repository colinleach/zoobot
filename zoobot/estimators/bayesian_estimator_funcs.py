import logging
import sys

import tensorflow as tf

from zoobot.estimators import losses

def estimator_wrapper(features, labels, mode, params):
    # estimator model funcs are only allowed to have (features, labels, params) arguments
    # re-order the arguments internally to allow for the custom class to be passed around
    # params is really the model class
    return params.entry_point(features, labels, mode)  # must have exactly the args (features, labels)


class BayesianModel():

    def __init__(
            self,
            image_dim,
            calculate_loss,
            output_dim,
            learning_rate=0.001,
            optimizer=tf.compat.v1.train.AdamOptimizer,
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
        self.output_dim = output_dim  # n final neuron
        self.image_dim = image_dim
        # self.calculate_loss = calculate_loss # callable loss = calculate_loss(labels, predictions) (or can subclass)
        # self.optimizer = optimizer
        # self.learning_rate = learning_rate
        # self.conv1_filters = conv1_filters
        # self.conv1_kernel = conv1_kernel
        # self.conv2_filters = conv2_filters
        # self.conv2_kernel = conv2_kernel
        self.conv3_filters = conv3_filters  # actually useful for the reshape
        # self.conv3_kernel = conv3_kernel
        # self.dense1_units = dense1_units
        # self.dense1_dropout = dense1_dropout
        # self.conv1_activation = conv1_activation
        # self.conv2_activation = conv2_activation
        # self.conv3_activation = conv3_activation
        # self.dense1_activation = dense1_activation
        # self.predict_dropout = predict_dropout  # dropout rate for predict mode
        # self.regression = regression
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


    def __call__(self, x, training):

        x = self.conv1(x)
        x = self.drop1(x, training=training)
        x = self.conv1b(x)
        x = self.drop1b(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.drop2(x, training=training)
        x = self.conv2b(x)
        x = self.drop2b(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.drop3(x, training=training)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.drop4(x, training=training)
        x = self.pool4(x)

        """
        Flatten tensor into a batch of vectors
        Start with image_dim shape. NB, does not change with channels: just alters num of first filters.
        2 * 2 * 2 = 8 factor reduction in shape from pooling, assuming stride 2 and pool_size 2
        length ^ 2 to make shape 1D
        64 filters in final layer
        """
        x = tf.reshape(x, [-1, int(self.image_dim / 16) ** 2 * self.conv3_filters], name='model/layer4/flat')

        x = self.dropout_final(x, training=training)
        x = self.dense_final(x)
        tf.summary.histogram('prediction', x)

        # normalise predictions by question (TODO hardcoded!)
        x = tf.concat([ tf.nn.softmax(x[:, :2]), tf.nn.softmax(x[:, 2:]) ], axis=1)
        tf.compat.v1.summary.histogram('normalised_prediction', x)
        return x


    # TODO move to shared utilities
    # TODO duplicated with input_utils
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])


    # def main_estimator(self, features, labels, mode):
    #     """
    #     Estimator wrapper function for four-layer cnn performing classification or regression
    #     Shows the general actions for each Estimator mode
    #     Details (e.g. neurons, activation funcs, etc) controlled by 'params'

    #     Args:
    #         features (tf.constant): images, shape (batch, x, y, 1)
    #         labels (tf.constant): labels, shape (batch, label_col)
    #         mode ():

    #     Returns:

    #     """
    #     if labels is not None:
    #         for n in range(self.output_dim):
    #             tf.summary.histogram('labels_{}'.format(n), labels[:, n])

    #     # TODO check args
    #     response, loss = self.bayesian_regressor(features, labels, mode)
        
        # TODO no longer needed?
        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     with tf.compat.v1.variable_scope('predict'):
        #         export_outputs = {
        #             signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(response)
        #         }
        #     return tf.estimator.EstimatorSpec(mode=mode, predictions=response, export_outputs=export_outputs)

        # assert labels is not None

        # if mode == tf.estimator.ModeKeys.TRAIN:
        #     with tf.compat.v1.variable_scope('train'):
        #         lr = tf.identity(self.learning_rate)
        #         tf.compat.v1.summary.scalar('learning_rate', lr)
        #         optimizer = self.optimizer(learning_rate=lr)

        #         # important to explicitly use within update_ops for batch norm to work
        #         # see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        #         update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        #         logging.warning(update_ops)
        #         with tf.control_dependencies(update_ops):
        #             train_op = optimizer.minimize(
        #                 loss=loss,
        #                 global_step=tf.compat.v1.train.get_global_step())
                
        #         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # else:  # must be EVAL mode
        #     with tf.compat.v1.variable_scope('eval'):
        #         # Add evaluation metrics (for EVAL mode)
        #         # eval_metric_ops = get_eval_metric_ops(self, labels, response)
        #         # return tf.estimator.EstimatorSpec(
        #         #     mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        #         eval_metric_ops = get_proxy_mean_squared_error_eval_ops(labels, response['prediction'])
        #         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



    # def bayesian_regressor(self, features, labels, mode):
    #     """
    #     Model function of four-layer CNN
    #     Can be used in isolation or called within an estimator e.g. four_layer_binary_classifier

    #     Args:
    #         features ():
    #         labels ():
    #         mode ():
    #         params ():

    #     Returns:

    #     """

    #     dropout_rate = self.dense1_dropout
    #     if mode == tf.estimator.ModeKeys.PREDICT:
    #         dropout_rate = self.predict_dropout

    #     # eval mode will have a lower loss than train mode, because dropout is off
    #     dropout_on = (mode == tf.estimator.ModeKeys.TRAIN) or (mode == tf.estimator.ModeKeys.PREDICT)
    #     tf.compat.v1.summary.scalar('dropout_on', tf.cast(dropout_on, tf.float32))
    #     tf.compat.v1.summary.scalar('dropout_rate', dropout_rate)

    #     dense1 = input_to_dense(features, mode, self)  # use batch normalisation
    #     predictions = dense_to_output(dense1, output_dim=self.output_dim, dropout_on=dropout_on, dropout_rate=dropout_rate)
    #     response = {'prediction': predictions}

    #     # if predict mode, feedforward from dense1 SEVERAL TIMES. Save all predictions under 'all_predictions'.
    #     if mode == tf.estimator.ModeKeys.PREDICT:
    #         return response, None  # no loss, as labels not known (in general)

    #     else: # calculate loss for TRAIN/EVAL with binomial
    #         # print_op = tf.print('labels', labels)
    #         # with tf.control_dependencies([print_op]):
    #         labels = tf.stop_gradient(labels)
    #         loss = self.calculate_loss(labels, predictions)
    #         mean_loss = tf.reduce_mean(input_tensor=loss)
    #         tf.compat.v1.losses.add_loss(mean_loss)
    #         return response, mean_loss


def get_proxy_mean_squared_error_eval_ops(labels, predictions):
    # TODO again, hardcoded!
    # smooth_observed_fracs = labels[:, :2]/tf.expand_dims(tf.reduce_sum(labels[:, :2], axis=1), axis=1)
    # spiral_observed_fracs = labels[:, 2:]/tf.expand_dims(tf.reduce_sum(labels[:, 2:], axis=1), axis=1)
    smooth_total = tf.reduce_sum(input_tensor=labels[:, :2], axis=1)
    spiral_total = tf.reduce_sum(input_tensor=labels[:, 2:], axis=1)
    tf.compat.v1.summary.histogram('smooth_total', smooth_total)
    tf.compat.v1.summary.histogram('spiral_total', spiral_total)
    smooth_observed_fracs = labels[:, 0]/smooth_total
    spiral_observed_fracs = labels[:, 2]/spiral_total
    # observed_vote_fractions = tf.concat([ labels[:, :2]/tf.expand_dims(tf.reduce_sum(labels[:, :2], axis=1), axis=1), labels[:, 2:]/tf.expand_dims(tf.reduce_sum(labels[:, 2:], axis=1), axis=1) ], axis=1)
    tf.compat.v1.summary.histogram('smooth_observed_fracs', smooth_observed_fracs)
    tf.compat.v1.summary.histogram('spiral_observed_fracs', spiral_observed_fracs)
    return {
        # "rmse": tf.metrics.root_mean_squared_error(observed_vote_fractions, predictions)
            'smooth_observed_fracs_eval': tf.compat.v1.metrics.root_mean_squared_error(smooth_observed_fracs, predictions[:, 0]),
            'spiral_observed_fracs_eval': tf.compat.v1.metrics.root_mean_squared_error(spiral_observed_fracs, predictions[:, 2])
        }

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
