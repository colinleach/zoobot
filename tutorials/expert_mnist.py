import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
# sess = tf.InteractiveSession()
#
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
#
# # run graph to the point where variables are initialised
# sess.run(tf.global_variables_initializer())
#
# y = tf.matmul(x,W) + b
#
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# # reduce in the sense of 'make one number' not 'decrease'
#
# # it's this line that does the reducing - optimizer.minimise
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# for _ in range(1000):
#   batch = mnist.train.next_batch(100)  # load another batch with helper function
#   train_step.run(feed_dict={x: batch[0], y_: batch[1]})  # can call optimizer.run()


# variable which will have random initial value, useful for weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# variable which will have slightly positive initial value of 0.1, useful for bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# do 2d convolution, given input x and weights w. Stride of 1, zero-padded (same output size)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # predefined convolution operation


# similarly for pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# shape of [patch size x2, input channels, output channels], all random values
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# put MNIST image back in natural shape, with first dim whatever size is required
x_image = tf.reshape(x, [-1, 28, 28, 1])

# apply relu function to the convolution
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer shape/inital values
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# flatten C output
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# apply the densely connected layer
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# placeholder lets us supply value - we can choose 0.5 for training and 1 for testing, for example
keep_prob = tf.placeholder(tf.float32)
# apply dropout to this FC layer - nn.dropout handles scaling
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
