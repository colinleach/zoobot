import pytest

from zoobot.active_learning import default_estimator_params
from zoobot.active_learning import create_instructions

def test_get_run_config():
    params = create_instructions.TrainCallableFactory(initial_size=28, final_size=14,  warm_start=True, test=False)  # not mocked but does very little anyway?
    log_dir = 'log_dir'
    train_records = ['something.tfrecord']
    eval_records = ['something_else.tfrecord']
    learning_rate = 0.01
    epochs = 5
    label_cols = ['label_a', 'label_b']
    default_estimator_params.get_run_config(params, log_dir, train_records, eval_records, learning_rate, epochs, label_cols)
