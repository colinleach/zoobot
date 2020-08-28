# import logging
# import sys

# import tensorflow as tf

# from zoobot.estimators import losses, efficientnet, custom_layers

# # def get_conv_drop_pool_block(filters, kernel, padding, activation, regularizer, )
# # https://www.tensorflow.org/guide/keras/overview#model_subclassing
# # https://www.tensorflow.org/guide/keras/custom_layers_and_models#building_models
# def get_bayesian_model(
#             image_dim,
#             output_dim,
#             schema,
#             conv1_filters=32,
#             conv1_kernel=1,
#             conv1_activation=tf.nn.relu,
#             conv2_filters=32,
#             conv2_kernel=3,
#             conv2_activation=tf.nn.relu,
#             conv3_filters=16,
#             conv3_kernel=3,
#             conv3_activation=tf.nn.relu,
#             dense1_units=128,
#             dense1_dropout=0.5,
#             dense1_activation=tf.nn.relu,
#             predict_dropout=0.5,
#             regression=False,
#             log_freq=10,    
#     ):

#         model = tf.keras.Sequential()

#         dropout_rate = 0  # no dropout on conv layers
#         regularizer = None
#         padding = 'same'
#         pool1_size = 2
#         pool1_strides = 2
#         pool2_size = 2
#         pool2_strides = 2
#         pool3_size = 2
#         pool3_strides = 2

#         model.add(tf.keras.layers.Convolution2D(
#             filters=conv1_filters,
#             kernel_size=[conv1_kernel, conv1_kernel],
#             padding=padding,
#             activation=conv1_activation,
#             kernel_regularizer=regularizer,
#             name='model/layer1/conv1')
#         )
#         # model.add(custom_layers.PermaDropout(rate=dropout_rate))
#         model.add(tf.keras.layers.Convolution2D(
#             filters=conv1_filters,
#             kernel_size=[conv1_kernel, conv1_kernel],
#             padding=padding,
#             activation=conv1_activation,
#             kernel_regularizer=regularizer,
#             name='model/layer1/conv1b')
#         )
#         # model.add(custom_layers.PermaDropout(rate=dropout_rate))
#         model.add(tf.keras.layers.MaxPooling2D(
#             pool_size=[pool1_size, pool1_size],
#             strides=pool1_strides,
#             name='model/layer1/pool1'))

#         model.add(tf.keras.layers.Convolution2D(
#             filters=conv2_filters,
#             kernel_size=[conv2_kernel, conv2_kernel],
#             padding=padding,
#             activation=conv2_activation,
#             kernel_regularizer=regularizer,
#             name='model/layer2/conv2')
#         )
#         # model.add(custom_layers.PermaDropout(rate=dropout_rate))
#         model.add(tf.keras.layers.Convolution2D(
#             filters=conv2_filters,
#             kernel_size=[conv2_kernel, conv2_kernel],
#             padding=padding,
#             activation=conv2_activation,
#             kernel_regularizer=regularizer,
#             name='model/layer2/conv2b')
#         )
#         # model.add(custom_layers.PermaDropout(rate=dropout_rate))
#         model.add(tf.keras.layers.MaxPooling2D(
#             pool_size=pool2_size,
#             strides=pool2_strides,
#             name='model/layer2/pool2')
#         )

#         model.add(tf.keras.layers.Convolution2D(
#             filters=conv3_filters,
#             kernel_size=[conv3_kernel, conv3_kernel],
#             padding=padding,
#             activation=conv3_activation,
#             kernel_regularizer=regularizer,
#             name='model/layer3/conv3')
#         )
#         # model.add(custom_layers.PermaDropout(rate=dropout_rate))
#         model.add(tf.keras.layers.MaxPooling2D(
#             pool_size=[pool3_size, pool3_size],
#             strides=pool3_strides,
#             name='model/layer3/pool3')
#         )

#         # identical to conv3
#         model.add(tf.keras.layers.Convolution2D(
#             filters=conv3_filters,
#             kernel_size=[conv3_kernel, conv3_kernel],
#             padding=padding,
#             activation=conv3_activation,
#             kernel_regularizer=regularizer,
#             name='model/layer4/conv4')
#         )
#         # model.add(custom_layers.PermaDropout(rate=dropout_rate))
#         model.add(tf.keras.layers.MaxPooling2D(
#             pool_size=[pool3_size, pool3_size],
#             strides=pool3_strides,
#             name='model/layer4/pool4')
#         )

#         # model.add(tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, int(image_dim / 16) ** 2 * conv3_filters], name='model/layer4/flat')))
#         model.add(tf.keras.layers.Flatten())

#         model.add(tf.keras.layers.Dense(
#             units=dense1_units,
#             activation=dense1_activation,
#             kernel_regularizer=regularizer,
#             name='model/layer4/dense1')
#         )
#         model.add(custom_layers.PermaDropout(rate=dense1_dropout))

#         # add to model
#         efficientnet.custom_top_dirichlet(model, output_dim, schema)

#         model.step = tf.Variable(0, dtype=tf.int64, name='model_step', trainable=False)  # will be updated by callback

#         return model


# # use any custom callback to keras.backend.set_value self.epoch=epoch, and then read that on each summary call
# # if this doesn't get train/test right, could similarly use the callbacks to set self.mode
# # https://www.tensorflow.org/guide/keras/custom_callback
# class UpdateStepCallback(tf.keras.callbacks.Callback):

#     def __init__(self, batch_size):
#         super(UpdateStepCallback, self).__init__()
#         self.batch_size = batch_size

#     def on_epoch_end(self, epoch, logs=None):
#         # print('\n\nStarting epoch', epoch, '\n\n')
#         # # access model with self.model, tf.ketas.backend.get/set_value
#         # # e.g. lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

#         # print('\n epoch ', epoch, type(epoch))
#         step = epoch * self.batch_size
#         # # self.model.step = step
#         # # self.model.step.assign(step)
#         tf.keras.backend.set_value(self.model.step, step)
#         print('\n Ending step: ', float(tf.keras.backend.get_value(self.model.step)))
#         # # print(f'Step {step}')
