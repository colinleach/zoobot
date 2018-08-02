import pytest

import tensorflow as tf
import numpy as np

from estimators import dummy_image_estimator


N_EXAMPLES = 2000


@pytest.fixture()
def features():
    # {'feature_name':array_of_values} format expected
    return {'x': np.random.rand(N_EXAMPLES, 28, 28, 1)}


@pytest.fixture()
def labels():
    return np.random.randint(low=0, high=2, size=N_EXAMPLES)


@pytest.fixture()
def batch_size():
    return 10


# https://www.tensorflow.org/get_started/datasets_quickstart
def train_input_fn(features, labels, batch_size):
    """An input function for training
    This input function builds an input pipeline that yields batches of (features, labels) pairs,
     where features is a dictionary features.
    """
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        print('No labels')
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples - don't repeat though!
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


# Build estimator from model function
@pytest.fixture()
def estimator(batch_size):

    return tf.estimator.Estimator(
        model_fn=dummy_image_estimator.dummy_model_fn,
        params={
            'batch_size': batch_size,  # required to get dimensions correct,
            'image_dim': 28
        }
        )


def test_training(estimator, features, labels, batch_size):
    # modifies estimator inplace
    estimator.train(
        input_fn=lambda: train_input_fn(features, labels, batch_size),
        steps=2
    )


def test_predict(estimator, features, labels, batch_size):
    test_training(estimator, features, labels, batch_size)  # requires model with a training
    predictions = estimator.predict(  # returns a generator
        input_fn=lambda: eval_input_fn(features, None, batch_size))
    print([predict for predict in predictions])


def test_eval(estimator, features, labels, batch_size):
    test_training(estimator, features, labels, batch_size)
    eval_result = estimator.evaluate(
        input_fn=lambda: train_input_fn(features, labels, batch_size),
        steps=1)  # or it never ends!
    print(eval_result)
