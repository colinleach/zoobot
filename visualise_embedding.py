import tensorflow as tf
import numpy as np

# from train_spiral_model import spiral_classifier
#
# builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel/')
#
# # Add graph and variables to builder and save
# with tf.Session() as sess:
#     builder.add_meta_graph_and_variables(
#         sess,
#         [tf.saved_model.tag_constants.TRAINING],
#         signature_def_map=None,
#         assets_collection=None)
# builder.save()
#
#
# def serving_input_receiver_fn():
#   """Build the serving inputs."""
#   # The outer dimension (None) allows us to batch up inputs for
#   # efficiency. However, it also means that if we want a prediction
#   # for a single instance, we'll need to wrap it in an outer list.
#   inputs = {"x": tf.placeholder(shape=[None, 4], dtype=tf.float32)}
#   return tf.estimator.export.ServingInputReceiver(inputs, inputs)
#
# export_dir = classifier.export_savedmodel(
#     export_dir_base="/path/to/model",
#     serving_input_receiver_fn=serving_input_receiver_fn)
#
#
# classifier.train(input_fn=train_input_fn, steps=2000)
# # [...snip...]
# predictions = list(classifier.predict(input_fn=predict_input_fn))
# predicted_classes = [p["classes"] for p in predictions]

# from tensorflow.contrib import predictor

saver = tf.train.import_meta_graph('/Data/repos/zoobot/runs/chollet_128/model.ckpt-271.meta')
graph = tf.get_default_graph()
# for n in graph.as_graph_def().node:
#     print('{}\n'.format(n.name))
dense1 = graph.get_tensor_by_name('model/layer3/dense1/kernel:0')
flat = graph.get_tensor_by_name('model/layer2/flat:0')


with tf.Session() as sess:

    saver.restore(sess, tf.train.latest_checkpoint('/data/repos/zoobot/runs/chollet_128/'))

    # dense1 = sess.run(dense1)
    # print(dense1)
    # print(type(dense1))
    # print(np.array(dense1.shape))

    flat = sess.run(flat)
    print(flat)
    print(type(flat))
    print(flat.shape)

    # print(sess.run(dense1_shape))
