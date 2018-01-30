import os

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from embeddings.chollet_128.make_sprites import images_to_sprite
from models.train_spiral_model import eval_input

# graph_to_import = '/Data/repos/zoobot/runs/chollet_128/model.ckpt-271.meta'
# variables_to_import = '/data/repos/zoobot/runs/chollet_128/'
graph_to_import = '/Data/repos/zoobot/runs/chollet_128_triple/model.ckpt-15901.meta'
variables_to_import = '/data/repos/zoobot/runs/chollet_128_triple/'

saver = tf.train.import_meta_graph(graph_to_import)
graph = tf.get_default_graph()
# for n in graph.as_graph_def().node:
#     print('{}\n'.format(n.name))

dense1 = graph.get_tensor_by_name('model/layer4/dense1/kernel:0')
flat = graph.get_tensor_by_name('model/layer2/flat:0')  # my own typo, is actually layer 3
# softmax = graph.get_tensor_by_name('model/layer3/softmax:0')
softmax = graph.get_tensor_by_name('softmax_tensor:0')

params = {}
params['image_dim'] = 128
params['batch_size'] = 128
params['eval_stratify'] = False  # something in stratify breaks with saved models

features, labels = eval_input(params)

# new_saver = tf.train.Saver(
#     {'images': features['x'],
#      'labels': labels,
#      'flat': flat})

# new_saver = tf.train.Saver(
#     {'flat': flat})


def save_labels_as_metadata(labels, save_loc):

    labels = [labels[n] for n in range(labels.shape[0])]
    data = [{'spiral': labels[n]} for n in range(len(labels))]
    metadata = pd.DataFrame(data)
    metadata.to_csv(save_loc, sep='\t')  # TensorFlow requires tab-separated


with tf.train.MonitoredSession() as sess:

    # fill in values
    saver.restore(sess, tf.train.latest_checkpoint(variables_to_import))

    # dense1 = sess.run(dense1)
    # print(dense1)
    # print(type(dense1))
    # print(np.array(dense1.shape))

    features, labels, flat_values, softmax_values, dense_values = sess.run([features, labels, flat, softmax, dense1])
    # print(features)
    # print(labels)
    print(labels.sum())
    # print(type(labels))
    # print(labels.shape)
    # print(flat)
    # print(type(flat))
    # print(flat_values.shape)
    print(softmax_values.shape)

    log_dir = '/data/repos/zoobot/embeddings/chollet_128'

    np.savetxt('{}/temp_flat_values.txt'.format(log_dir), flat_values)
    np.savetxt('{}/temp_softmax_values.txt'.format(log_dir), softmax_values)
    np.savetxt('{}/temp_dense_values.txt'.format(log_dir), dense_values)
    save_labels_as_metadata(labels, '{}/metadata.tsv'.format(log_dir))

    # https://github.com/tensorflow/tensorflow/issues/8425
    # saver.save(
    #     sess._sess._sess._sess._sess,
    #     '{}/model.ckpt'.format(log_dir)
    # )

    # Taken from: https://github.com/tensorflow/tensorflow/issues/6322
    sprite = images_to_sprite(features['x'], labels)
    scipy.misc.imsave('{}/sprite.png'.format(log_dir), sprite)

flat_values = np.loadtxt('{}/temp_flat_values.txt'.format(log_dir))
flat_variable = tf.Variable(initial_value=flat_values, dtype=tf.float32, name='flat')

softmax_values = np.loadtxt('{}/temp_softmax_values.txt'.format(log_dir))
softmax_variable = tf.Variable(initial_value=softmax_values, dtype=tf.float32, name='softmax')

dense_values = np.loadtxt('{}/temp_dense_values.txt'.format(log_dir))
dense_variable = tf.Variable(initial_value=dense_values, dtype=tf.float32, name='dense')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # breaks if run without some selections
    # saver = tf.train.Saver(
    #     {'flat': flat_variable,
    #      'softmax': softmax_variable})
    saver = tf.train.Saver(
        {'flat': flat_variable})
    saver.save(
        sess,
        '{}/model.ckpt'.format(log_dir)
    )

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.

    embedding = config.embeddings.add()
    embedding.tensor_name = flat_variable.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    # Comment out if you don't want sprites
    embedding.sprite.image_path = os.path.join(log_dir, 'sprite.png')
    embedding.sprite.single_image_dim.extend([128, 128])
    #
    # embedding = config.embeddings.add()
    # embedding.tensor_name = softmax_variable.name
    # # Link this tensor to its metadata file (e.g. labels).
    # embedding.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    # # Comment out if you don't want sprites
    # embedding.sprite.image_path = os.path.join(log_dir, 'sprite.png')
    # embedding.sprite.single_image_dim.extend([128, 128])

    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)
