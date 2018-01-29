import tensorflow as tf

from models.train_spiral_model import spiral_classifier, run_experiment


def default_params():
    return dict(
        epochs=1000,
        batch_size=128,
        image_dim=64,
        train_batches=30,
        eval_batches=3,
        max_train_batches=None,
        log_freq=25,
        train_stratify=True,
        eval_stratify=True
)


def default_model_architecture():
    return dict(
        padding='same',

        conv1_filters=32,
        conv1_kernel=[5, 5],

        conv1_activation=tf.nn.relu,

        pool1_size=[2, 2],
        pool1_strides=2,

        conv2_filters=64,
        conv2_kernel=[5, 5],
        conv2_activation=tf.nn.relu,

        pool2_size=[2, 2],
        pool2_strides=2,

        dense1_units=1064,
        dense1_dropout=0.4,
        dense1_activation=tf.nn.relu,

        learning_rate=0.001,
        optimizer=tf.train.GradientDescentOptimizer,

        log_dir='runs/default_run'

    )


def chollet_model_architecture():
    return dict(
        padding='same',

        conv1_filters=32,
        conv1_kernel=[3, 3],

        conv1_activation=tf.nn.relu,

        pool1_size=[2, 2],
        pool1_strides=2,

        conv2_filters=32,
        conv2_kernel=[3, 3],
        conv2_activation=tf.nn.relu,

        pool2_size=[2, 2],
        pool2_strides=2,

        conv3_filters=64,
        conv3_kernel=[3, 3],
        conv3_activation=tf.nn.relu,

        pool3_size=[2, 2],
        pool3_strides=2,

        dense1_units=1064,
        dense1_dropout=0.5,
        dense1_activation=tf.nn.relu,

        learning_rate=0.001,
        optimizer=tf.train.GradientDescentOptimizer,


    )


if __name__ == '__main__':
    params = default_params()
    params.update(chollet_model_architecture())
    params['image_dim'] = 128
    params['log_dir'] = 'runs/chollet_128_triple'
    run_experiment(spiral_classifier, params)
