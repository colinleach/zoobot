import tensorflow as tf

from zoobot.catalogs.tfrecord.read_tfrecord import read_and_decode_single_example

# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

filename = '/data/galaxy_zoo/gz2/tfrecord/spiral_28.tfrecord'
label, image, spiral_fraction = read_and_decode_single_example(filename)

images_batch, labels_batch, spiral_fraction_batch = tf.train.shuffle_batch(
    [image, label, spiral_fraction], batch_size=10,
    capacity=100000,
    min_after_dequeue=1000)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
labels, images, spiral_fractions = sess.run([labels_batch, images_batch, spiral_fraction_batch])

square_images = images.reshape([-1, 28, 28, 3])
input_layer = tf.reduce_mean(square_images, axis=3, keep_dims=True)
print(input_layer.shape)


# Convolutional Layer #1
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
# Dense Layer
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(
    inputs=dense, rate=0.4)
# Logits Layer
logits = tf.layers.dense(inputs=dropout, units=2)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

# for monitoring
loss_mean = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

while True:
  _, loss_val = sess.run([train_op, loss_mean])
  print(loss_val)