import logging
import sys

import tensorflow as tf

from zoobot.estimators import losses

# https://www.tensorflow.org/guide/keras/overview#model_subclassing
# https://www.tensorflow.org/guide/keras/custom_layers_and_models#building_models
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
        self.schema = schema
        self.conv3_filters = conv3_filters  # actually useful elsewhere for the reshape
        self.log_freq = log_freq
        # self.logging_hooks = logging_hooks(self)  # TODO strange error with passing this to estimator in params
        self.logging_hooks = [None, None, None]
        # self.step = tf.Variable(0, dtype=tf.int64, name='model_step', trainable=False)  # will be updated by callback
 
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
        self.dropout_final = tf.keras.layers.Dropout(rate=dropout_rate)  # possible massive typo - this has no dropout!
        self.dense_final = tf.keras.layers.Dense(
            units=self.output_dim,  # num outputs
            name='model/layer5/dense1')

    def build(self, input_shape):
        # self.step = 0
        self.step = tf.Variable(0, dtype=tf.int64, name='model_step', trainable=False)  # will be updated by callback

        super().build(input_shape)

    # really important to use call, not __call__, or the dataset won't be made in graph mode and you'll get eager/symbolic mismatch error
    # do not decorate, it messes up the tf.summary writing. Keras will automatically convert to graph mode anyway
    # https://github.com/tensorflow/tensorflow/issues/32889
    def call(self, x, training=None):

        # x = x / 256.  # rescale 0->1 from 0->256

        tf.summary.image('input_image', x, step=self.step)
        tf.summary.histogram('input_image_hist', x, step=self.step)

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

        # normalise predictions by question
        # list comp is allowed since schema is pure python, not tensors, but note that it must be static or graph will be wrong
        x = tf.concat([tf.nn.softmax(x[:, q[0]:q[1]+1]) for q in self.schema.question_index_groups], axis=1)
        # tf.summary.histogram('x', x, step=self.step)
        for q, (start_index, end_index) in self.schema.named_index_groups.items():
            for i in range(start_index, end_index+1):
                tf.summary.histogram(f'prediction_{self.schema.label_cols[i]}', x[:, i], step=self.step)

        return x


    # TODO move to shared utilities
    # TODO duplicated with input_utils
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])

@tf.function
def squared_error(labels, predictions):
    total = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    observed_fracs = tf.math.divide_no_nan(labels, total)  # WARNING need to check into this
    sq_error = (observed_fracs - predictions) ** 2
    return sq_error

@tf.function
def absolute_error(labels, predictions):
    total = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    observed_fracs = tf.math.divide_no_nan(labels, total)  # WARNING need to check into this
    abs_error = tf.math.abs(observed_fracs - predictions)
    return abs_error

class CustomLossByQuestion(tf.keras.metrics.Metric):

    def __init__(self, name, start_col, end_col, **kwargs):
        super(CustomLossByQuestion, self).__init__(name=name, **kwargs)
        self.loss = self.add_weight(name=name, initializer='zeros')
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        self.total = tf.Variable(0., dtype=tf.float32)
        self.start_col = start_col
        self.end_col = end_col

    def update_state(self, y_true, y_pred, sample_weight=None):
        q_loss = losses.multinomial_loss(y_true[:, self.start_col:self.end_col+1], y_pred[:, self.start_col:self.end_col+1])
        self.total.assign_add(tf.reduce_sum(q_loss))
        self.batch_count.assign_add(tf.shape(y_true)[0])
        self.loss.assign(self.total/tf.cast(self.batch_count, dtype=tf.float32))

    def result(self):
        return self.loss

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.loss.assign(0.)
        self.batch_count.assign(0)
        self.total.assign(0.)  

class CustomLossByAnswer(tf.keras.metrics.Metric):

    def __init__(self, name, col, **kwargs):
        super(CustomLossByAnswer, self).__init__(name=name, **kwargs)
        self.loss = self.add_weight(name=name, initializer='zeros')
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        self.total = tf.Variable(0., dtype=tf.float32)
        self.col = col

    def update_state(self, y_true, y_pred, sample_weight=None):
        expected_probs = y_pred[self.col]
        successes = y_true[self.col]
        a_loss = -successes * tf.math.log(expected_probs + tf.constant(1e-8, dtype=tf.float32))
        self.total.assign_add(tf.reduce_sum(a_loss))
        self.batch_count.assign_add(tf.shape(y_true)[0])
        self.loss.assign(self.total/tf.cast(self.batch_count, dtype=tf.float32))

    def result(self):
        return self.loss

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.loss.assign(0.)
        self.batch_count.assign(0)
        self.total.assign(0.)  


class CustomAbsErrorByColumn(tf.keras.metrics.Metric):

    def __init__(self, name, start_col, end_col, **kwargs):
        print(f'Name: {name}, start {start_col}, end {end_col}')
        super(CustomAbsErrorByColumn, self).__init__(name=name, **kwargs)
        self.abs_error = self.add_weight(name=name, initializer='zeros')
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        self.total = tf.Variable(0., dtype=tf.float32)
        self.start_col = start_col
        self.end_col = end_col

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total.assign_add(tf.reduce_sum(absolute_error(y_true[:, self.start_col:self.end_col+1], y_pred[:, self.start_col:self.end_col+1])))
        self.batch_count.assign_add(tf.shape(y_true)[0])
        self.abs_error.assign(self.total/tf.cast(self.batch_count, dtype=tf.float32))

    def result(self):
        return self.abs_error

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.abs_error.assign(0.)
        self.batch_count.assign(0)
        self.total.assign(0.)  



# use any custom callback to keras.backend.set_value self.epoch=epoch, and then read that on each summary call
# if this doesn't get train/test right, could similarly use the callbacks to set self.mode
# https://www.tensorflow.org/guide/keras/custom_callback
class UpdateStepCallback(tf.keras.callbacks.Callback):

    def __init__(self, batch_size):
        super(UpdateStepCallback, self).__init__()
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        # print('\n\nStarting epoch', epoch, '\n\n')
        # # access model with self.model, tf.ketas.backend.get/set_value
        # # e.g. lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        # print('\n epoch ', epoch, type(epoch))
        step = epoch * self.batch_size
        # # self.model.step = step
        # # self.model.step.assign(step)
        tf.keras.backend.set_value(self.model.step, step)
        print('\n Ending step: ', float(tf.keras.backend.get_value(self.model.step)))
        # # print(f'Step {step}')


# def get_gz_binomial_eval_metric_ops(self, y_true, predictions):
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
