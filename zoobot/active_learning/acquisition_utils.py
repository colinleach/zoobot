import os
import statistics  # thanks Python 3.4!

import numpy as np
import matplotlib
from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from shared_astro_utils import plotting_utils
from zoobot.estimators import make_predictions


def get_mean_k_predictions(binomial_probs_per_sample):
    # average over samples to get the mean prediction per subject (0th), per k (1st)
    mean_predictions = []
    for galaxy in binomial_probs_per_sample:
        all_samples = np.stack([sample for sample in galaxy])
        mean_prediction = np.mean(all_samples, axis=0)  # 0th axis is sample, 1st is k
        mean_predictions.append(mean_prediction)
    return mean_predictions


def binomial_entropy(rho, n_draws):
    """
    If possible, calculate bin probs only once, for speed
    Only use this function when rho is only used here
    """
    binomial_probs = make_predictions.binomial_prob_per_k(rho, n_draws)
    return distribution_entropy(binomial_probs)
binomial_entropy = np.vectorize(binomial_entropy)


def distribution_entropy(probabilities):
    try:
        assert isinstance(probabilities, np.ndarray)  # e.g. array of p(k|n) for many k, one subject
        assert probabilities.ndim == 1
        assert probabilities.max() <= 1. 
        assert probabilities.min() >= 0.
    except:
        print(probabilities)
        print(type(probabilities))
        raise ValueError('Probabilities must be ndarray of values between 0 and 1')
    return float(
        np.sum(
            list(map(
                lambda p: -p * np.log(p + 1e-12),
                probabilities)
            )
        )
    )

def predictive_binomial_entropy(bin_probs_of_samples):
    """[summary]
    
    Args:
        sampled_rho (float): MLEs of binomial probability, of any dimension
        n_draws (int): N draws for those MLEs.
    
    Returns:
        (float): entropy of binomial with N draws and p=sampled rho, same shape as inputs
    """
    # average over samples to get the mean prediction per k, per subject
    mean_k_predictions = get_mean_k_predictions(bin_probs_of_samples)
    # get the distribution entropy for each of those mean predictions, return as array
    return np.array([distribution_entropy(probabilities) for probabilities in mean_k_predictions])


def expected_binomial_entropy(bin_probs_of_samples):
    # get the entropy over all k (reduce axis 2)
    n_subjects, n_samples = len(bin_probs_of_samples), len(bin_probs_of_samples[0])
    binomial_entropy_of_samples = np.zeros((n_subjects, n_samples))
    for subject_n in range(n_subjects):
        for sample_n in range(n_samples):
            entropy_of_sample = distribution_entropy(bin_probs_of_samples[subject_n][sample_n])
            binomial_entropy_of_samples[subject_n, sample_n] = entropy_of_sample
    # average over samples (reduce axis 1)
    return np.mean(binomial_entropy_of_samples, axis=1) 


# new kde mutual info func, requires genuinely several models
def multimodel_bald(model_predictions, min_entropy=-5.5, max_entropy=10):
    individual_entropies = np.array([get_kde(predictions).entropy for predictions in model_predictions])
    mean_individual_entropy = individual_entropies.mean()
    expected_entropy = get_kde(model_predictions.flatten()).entropy
    if individual_entropies.min() < min_entropy:  # occasionally, very nearby predictions can cause insanely spiked kdes
        return np.nan
    elif individual_entropies.max() > max_entropy:
        return np.nan
    else:
        return expected_entropy - mean_individual_entropy


def calculate_reliable_multimodel_mean_acq(samples_list, schema):
    acq = get_multimodel_acq(samples_list)
    acq_with_nans = sense_check_multimodel_acq(acq, schema)
    mean_acq = np.mean(acq_with_nans, axis=1)
    mean_acq[np.isnan(mean_acq)] = -99.  # never acquire these. nan are a pain for argsort and db.
    return mean_acq


def get_multimodel_acq(samples_list, n_processes=8):  # e.g. [samples_a, samples_b], each of shape(galaxy, answer, rho)
    samples = np.stack(samples_list, axis=-1)  # add new final axis, which model
    n_subjects = samples.shape[0]
    n_answers = samples.shape[1]
    acq = np.zeros((n_subjects, n_answers))
    for subject_n in range(n_subjects):
        model_predictions_by_answer = [samples[subject_n, answer_n].transpose() for answer_n in range(n_answers)]  # shape (model, rho)
        with Pool(processes=n_processes) as pool:
            acq_by_answer = pool.map(multimodel_bald, model_predictions_by_answer)
        acq[subject_n, :] = acq_by_answer
    return acq


def get_kde(x):
    estimator = sm.nonparametric.KDEUnivariate(x)
    estimator.fit(fft=True)
    return estimator


def sense_check_multimodel_acq(acq, schema, min_acq=-0.5):
    assert acq.shape[1] == len(schema.answers)
    equal_binary_responses = [check_equal_binary_responses(row, schema) for row in acq]
    above_min = np.all(acq > min_acq, axis=1)
    acq[~equal_binary_responses | ~above_min] = np.nan  # set row to nan to preserve order
    return acq  # be sure to remove nans before applying argsort!


def check_equal_binary_responses(acq_row, schema):
    binary_questions = [q for q in schema.questions if len(q.answers) == 2]
    for q in binary_questions:
        indices = schema.named_index_groups[q]
        if not np.isclose(acq_row[indices[0]], acq_row[indices[1]], rtol=.01):
            return False
    return True




def mutual_info_acquisition_func_multiq(samples, schema, retirement=40):
    assert samples.ndim == 3  # batch, p, model
    all_expected_mi = []
    for q in schema.questions:
        prev_q = q.asked_after
        if prev_q is None:
            expected_votes = retirement
        else:
            joint_p_of_asked = schema.joint_p(samples.mean(axis=-1), prev_q.text)  # prob of getting the answer needed to ask this question
            expected_votes = joint_p_of_asked * retirement
        for a in q.answers:
            all_expected_mi.append(mutual_info_acquisition_func(samples[:, a.index], expected_votes=expected_votes))
    # return np.sum(all_expected_mi)
    return np.array(all_expected_mi).swapaxes(0, 1)  # keep (batch, answer) convention. Note: not yet summed over answers


def mutual_info_acquisition_func(samples: np.ndarray, expected_votes):
    """Calculate BALD based on binomial p's estimated from several 'models'
    
    Args:
        samples (np.ndarray): (batch, model)-shape bin probs. No answer dim!
        expected_votes ([type]): Use as N in bin probs. If int, will be copied across batch dimension.
    
    Returns:
        [type]: [description]
    """
    assert samples.ndim == 2  # no answer dim allowed!
    if isinstance(expected_votes, int):
        typical_votes = expected_votes
        expected_votes = [typical_votes for n in range(len(samples))]
    assert len(samples) == len(expected_votes)
    bin_probs = make_predictions.bin_prob_of_samples(samples, expected_votes)
    predictive_entropy = predictive_binomial_entropy(bin_probs)
    expected_entropy = expected_binomial_entropy(bin_probs)
    mutual_info = predictive_entropy - expected_entropy
    return [float(mutual_info[n]) for n in range(len(mutual_info))]  # return a list


def sample_variance(samples):
    """Mean deviation from the mean. Only meaningful for unimodal distributions.
    See http://mathworld.wolfram.com/SampleVariance.html
    
    Args:
        samples (np.array): predictions of shape (galaxy_n, sample_n)
    
    Returns:
        np.array: variance by galaxy, of shape (galaxy_n)
    """

    return np.apply_along_axis(statistics.variance, arr=samples, axis=1)


def show_acquisitions_from_tfrecords(tfrecord_locs, predictions, acq_string, save_dir):
    """[summary]
    
    Args:
        tfrecord_locs ([type]): [description]
        predictions ([type]): [description]
        acq_string ([type]): [description]
        save_dir ([type]): [description]
    """
    raise NotImplementedError
    # subjects = get_subjects_from_tfrecords_by_id_str(tfrecord_locs, id_strs)
    # images = [subject['matrix'] for subject in subjects]
    # save_acquisition_examples(images, predictions.acq_values, acq_string, save_dir)


def save_acquisition_examples(images, acq_values, acq_string, save_dir):
    """[summary]
    
    Args:
        images (np.array): of form [n_subjects, height, width, channels]. NOT a list.
        acq_values ([type]): [description]
        acq_string ([type]): [description]
        save_dir ([type]): [description]
    """
    assert isinstance(images, np.ndarray)
    assert isinstance(acq_values, np.ndarray)
    # show galaxies with max/min variance, or top/bottom 20% of variance (more representative)
    sorted_galaxies = images[acq_values.argsort()]
    min_gals = sorted_galaxies
    max_gals = sorted_galaxies[::-1]  # reverse
    low_galaxies = sorted_galaxies[:int(len(images)/5.)]
    high_galaxies = sorted_galaxies[int(-len(images)/5.):]
    np.random.shuffle(low_galaxies)   # inplace
    np.random.shuffle(high_galaxies)  # inplace

    galaxies_to_show = [
        {
            'galaxies': min_gals, 
            'save_loc': os.path.join(save_dir, 'min_{}.png'.format(acq_string))
        },
        {
            'galaxies': max_gals,
            'save_loc': os.path.join(save_dir, 'max_{}.png'.format(acq_string))
        },
        {
            'galaxies': high_galaxies,
            'save_loc': os.path.join(save_dir, 'high_{}.png'.format(acq_string))
        },
        {
            'galaxies': low_galaxies,
            'save_loc': os.path.join(save_dir, 'low_{}.png'.format(acq_string))
        },
    ]

    # save images
    for galaxy_set in galaxies_to_show:
        assert len(galaxy_set['galaxies']) != 0
        plotting_utils.plot_galaxy_grid(
            galaxies=galaxy_set['galaxies'],
            rows=9,
            columns=3,
            save_loc=galaxy_set['save_loc']
        )
