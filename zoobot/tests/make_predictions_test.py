import pytest

from zoobot.estimators import make_predictions

def test_load_predictor(predictor_model_loc):
    predictor = make_predictions.load_predictor(predictor_model_loc)
    assert callable(predictor)

def test_get_samples_of_subjects(predictor, parsed_example):
    n_subjects = 10
    n_samples = 5
    subjects = [parsed_example for n in range(n_subjects)]
    samples = make_predictions.get_samples_of_subjects(predictor, subjects, n_samples)
    assert samples.shape == (10, 5)
