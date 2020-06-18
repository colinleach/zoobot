import os
import statistics  # thanks Python 3.4!
import logging
import itertools
from functools import lru_cache

import numpy as np
import matplotlib
from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import tensorflow_probability as tfp
import tensorflow as tf

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
    logging.info('Using multi-model acquisition function')
    acq, _, _ = get_multimodel_acq(samples_list, schema)
    # acq_with_nans = sense_check_multimodel_acq(acq, schema)
    mean_acq = np.mean(acq, axis=1)
    mean_acq[np.isnan(mean_acq)] = -99.  # never acquire these. nan are a pain for argsort and db.
    return mean_acq


def get_multimodel_acq(samples_list, schema, retirement=40):  # e.g. [samples_a, samples_b], each of shape(galaxy, answer, rho)
    joint_samples = np.stack(samples_list, axis=-1)  # add new final axis, which model
    n_subjects = joint_samples.shape[0]
    n_answers = joint_samples.shape[1]
    logging.info('Beginning acquisition calculations ({} models, {} subjects, {} answers)'.format(len(samples_list), n_subjects, n_answers))
    
    # binomial mode
    # acq = np.zeros((n_subjects, n_answers))
    # for subject_n in range(n_subjects):
    #     model_predictions_by_answer = [samples[subject_n, answer_n].transpose() for answer_n in range(n_answers)]  # shape (model, rho)
    #     with Pool(processes=n_processes) as pool:
    #         acq_by_answer = pool.map(multimodel_bald, model_predictions_by_answer)
    #     acq[subject_n, :] = acq_by_answer

    # dirichlet mode
    all_expected_mi = []
    all_predictive_entropy = []
    all_expected_entropy = []

    # used for working out prob q's are asked
    joint_samples = tf.concat(samples_list, axis=2) 
    prob_of_answers = dirichlet_prob_of_answers(joint_samples, schema)

    for q in schema.questions:
        if q.text == 'spiral-count':
            # should be save to skip as I immediately take an average
            logging.warning('Temporarily skipping spiral-count as too many answers')
        else:
            print(q.text)
            expected_votes_list = [get_expected_votes(samples, q, retirement, schema, round=True) for samples in samples_list]
            # print(expected_votes_list)

            samples_list_by_q = [samples[:, q.start_index:q.end_index+1] for samples in samples_list]
            print('Calculating predictive entropy')
            predictive_entropy = dirichlet_predictive_entropy(samples_list_by_q, expected_votes_list)
            print('Calculating expected entropy')
            expected_entropy = dirichlet_expected_entropy(samples_list_by_q, expected_votes_list)
            mi_for_q = predictive_entropy - expected_entropy
            prev_q = q.asked_after
            if prev_q is None:
                joint_p_of_asked = 1.
            else:
                joint_p_of_asked = schema.joint_p(prob_of_answers, q.asked_after.text)
            # print(mi_for_q)

            all_predictive_entropy.append(predictive_entropy)
            all_expected_entropy.append(expected_entropy)
            all_expected_mi.append(mi_for_q * joint_p_of_asked)


    logging.info('Acquisition calculations complete')
    return np.array(all_expected_mi).transpose(), np.array(all_predictive_entropy).transpose(), np.array(all_expected_entropy).transpose()


def get_expected_votes(samples, q, retirement, schema, round):
    prob_of_answers = dirichlet_prob_of_answers(samples, schema)  # mean over both models. Prob given q is asked!
    prev_q = q.asked_after
    if prev_q is None:
        expected_votes = tf.ones(len(samples)) * retirement
    else:
        joint_p_of_asked = schema.joint_p(prob_of_answers, prev_q.text)  # prob of getting the answer needed to ask this question
        expected_votes = joint_p_of_asked * retirement
    if round:
        return tf.round(expected_votes)
    else:
        return expected_votes


def get_kde(x):
    estimator = sm.nonparametric.KDEUnivariate(x)
    estimator.fit(fft=True)
    return estimator


def sense_check_multimodel_acq(acq, schema, min_acq=-0.5):
    assert acq.shape[1] == len(schema.answers)
    equal_binary_responses = np.array([check_equal_binary_responses(row, schema) for row in acq])
    above_min = np.all(acq > min_acq, axis=1)
    acq_is_sensible = equal_binary_responses & above_min
    logging.info('{} sensible acq values ({} equal binary, {} above min {})'.format(acq_is_sensible.mean(), equal_binary_responses.mean(), above_min.mean(), min_acq))
    acq[~acq_is_sensible] = np.nan  # set row to nan to preserve order
    return acq  # be sure to remove nans before applying argsort!


def check_equal_binary_responses(acq_row, schema):
    binary_questions = [q for q in schema.questions if len(q.answers) == 2]
    for q in binary_questions:
        indices = schema.named_index_groups[q]
        if not np.isclose(acq_row[indices[0]], acq_row[indices[1]], rtol=.01):
            return False
    return True

# for one model...
# def mutual_info_acquisition_func_multiq(samples, schema, retirement=40):
#     assert samples.ndim == 3  # batch, p, model
#     all_expected_mi = []
#     for q in schema.questions:
#         prev_q = q.asked_after
#         if prev_q is None:
#             expected_votes = retirement
#         else:
#             # binomial mode
#             # prob_of_answers = binomial_prob_of_answers(samples)
#             # or dirichlet mode
#             prob_of_answers = dirichlet_prob_of_answers(samples, schema)
#             joint_p_of_asked = schema.joint_p(prob_of_answers, prev_q.text)  # prob of getting the answer needed to ask this question
#             expected_votes = joint_p_of_asked * retirement

#         # binomial mode
#         # for a in q.answers:
#         #     all_expected_mi.append(mutual_info_acquisition_func(samples[:, a.index], expected_votes=expected_votes))
#         # OR dirichlet mode
#         all_expected_mi.append(dirichlet_mutual_information(samples[:, q.start_index:q.end_index+1], expected_votes))

#     return np.array(all_expected_mi).swapaxes(0, 1)  # keep (batch, answer) convention. Note: not yet summed over answers. Note: dirichlet gives (batch, question) not (batch, answer) - better!


def binomial_prob_of_answers(samples):
    # samples has (batch, answer, dropout) shape
    return samples.mean(axis=-1)

def dirichlet_prob_of_answers(samples, schema):
    # mean probability (including dropout) of an answer being given. 
    # samples has (batch, answer, dropout) shape
    p_of_answers = []
    for q in schema.questions:
        samples_by_q = samples[:, q.start_index:q.end_index+1, :]
        p_of_answers.append(dirichlet_mixture(samples_by_q, expected_votes=1, n_samples=samples.shape[2]).mean().numpy())
    # print([p.shape for p in p_of_answers])
    p_of_answers = np.concatenate(p_of_answers, axis=1)
    return p_of_answers


# def dirichlet_mutual_information(samples_for_q, expected_votes):
#     # samples_for_q has shape (batch, concentration, dropout)
#     # MI = entropy over all samples - mean entropy of each sample

#     # predictive_entropy = dirichlet_mixture(samples_for_q, expected_votes).entropy.numpy()  # or .entropy_lower_bound
#     predictive_entropy = dirichlet_predictive_entropy(samples_for_q, expected_votes)
#     expected_entropy = dirichlet_expected_entropy(samples_for_q, expected_votes)
#     return predictive_entropy - expected_entropy  # (batch) shape



def dirichlet_expected_entropy(list_of_samples_for_q, list_of_expected_votes):
    expected_entropies = []
    for n, samples_for_q in enumerate(list_of_samples_for_q):  # for each model
        expected_votes = list_of_expected_votes[n]
        # predictive entropy per model is an expected entropy for one model (now over dropout)
        expected_entropies.append(dirichlet_predictive_entropy_per_model(samples_for_q, expected_votes))
    return np.mean(expected_entropies, axis=0)


def dirichlet_predictive_entropy(list_of_samples_for_q, list_of_expected_votes):
    # print(list_of_samples_for_q[0].shape)
    joint_samples = tf.concat(list_of_samples_for_q, axis=2)  # concat along dropout (sample) axis
    stacked_expected_votes = tf.stack(list_of_expected_votes, axis=1)
    # print(stacked_expected_votes)
    joint_expected_votes = tf.round(tf.reduce_mean(stacked_expected_votes, axis=1))  # take the average expected votes. But won't be int?
    # now we consider these samples from one model
    # print(joint_expected_votes)
    return dirichlet_predictive_entropy_per_model(joint_samples, joint_expected_votes)


def dirichlet_expected_entropy_per_model(samples_for_q, expected_votes):
    assert tf.rank(samples_for_q) == 3
    n_answers = samples_for_q.shape[1]
    n_samples = samples_for_q.shape[2]
    expected_entropy = []
    assert tf.rank(expected_votes) == 1
    for galaxy_n, galaxy in enumerate(samples_for_q):  # rows
        entropies_this_galaxy = []
        for dropout_n in range(n_samples):
            # print(galaxy_n, dropout_n)
            galaxy_with_dummy_batch = np.expand_dims(galaxy[:, dropout_n], axis=0)  # mixture requires batch dimension
            dist = tfp.distributions.DirichletMultinomial(expected_votes[galaxy_n], galaxy_with_dummy_batch, validate_args=True)
            entropies_this_galaxy.append(entropy_in_permutations_by_galaxy(dist, expected_votes[galaxy_n], n_answers))  # cannot have batch dimension > 1
        expected_entropy.append(np.mean(entropies_this_galaxy))
    return np.array(expected_entropy)


def dirichlet_predictive_entropy_per_model(samples_for_q, expected_votes):
    entropies = []
    n_answers = samples_for_q.shape[1]
    n_samples = samples_for_q.shape[2]
    for galaxy_n, galaxy in enumerate(samples_for_q):  # includes dropout
        galaxy_with_dummy_batch = np.expand_dims(galaxy, axis=0)  # mixture requires batch dimension
        # print(galaxy_n, expected_votes[galaxy_n], n_answers, n_samples)
        mixture = dirichlet_mixture(galaxy_with_dummy_batch, expected_votes[galaxy_n], n_samples)
        entropies.append(entropy_in_permutations_by_galaxy(mixture, expected_votes[galaxy_n], n_answers))
    return np.array(entropies)


def entropy_in_permutations_by_galaxy(dist, total_count, n_answers):
    assert dist.batch_shape == tf.TensorShape([])  or dist.batch_shape == tf.TensorShape([1]) # works on single galaxy
    assert dist.event_shape == tf.TensorShape([n_answers])  # need concentrations for every answer
    permutations = np.array(list(get_discrete_permutations(total_count, n_answers)))
    probs = dist.prob(permutations)
    return -np.sum(probs * np.log(probs))


# https://stackoverflow.com/questions/37711817/generate-all-possible-outcomes-of-k-balls-in-n-bins-sum-of-multinomial-catego
def get_discrete_permutations(n, k):
    if isinstance(n, np.ndarray):
        assert n.size == 1
    if isinstance(n, np.ndarray) or isinstance(n, np.float32):
        n = int(n)
    if not isinstance(n, int):  # must be tensor
        n = int(n.numpy())
    # print(type(k), type(n))
    def permutation_generator(n, k):
        masks = np.identity(k, dtype=int)
        for c in itertools.combinations_with_replacement(masks, n): 
            yield sum(c)
    return np.array(list(permutation_generator(n, k)))


def dirichlet_entropy_in_concentrations(samples_for_q, expected_votes):
    # samples should have shape (batch, answers_for_q), with values being the concentration of each galaxy/answer
    dist = tfp.distributions.DirichletMultinomial(expected_votes, samples_for_q, validate_args=True)
    return dist.entropy


def dirichlet_mixture(samples_for_q, expected_votes, n_samples):
    assert samples_for_q.ndim == 3  # must have galaxies dimension (:1 will work)
    # samples_for_q has shape (batch, answer, dropout)
    # n_samples = samples_for_q.shape[2]  # is a np array so this is okay

    component_probs = tf.zeros(n_samples) / n_samples
    categorical = tfp.distributions.Categorical(logits=component_probs, validate_args=True)

    # move dropout to (leading) batch dimensions, concentrations to (last) event dim. Now (batch, dropout, concentrations)
    concentrations = tf.transpose(samples_for_q, (0, 2, 1))
    # print(concentrations.shape)
    component_distribution = tfp.distributions.DirichletMultinomial(expected_votes, concentrations, validate_args=True)
    # print(component_distribution)

    mixture = tfp.distributions.MixtureSameFamily(
        categorical,
        component_distribution,
        validate_args=True
    )
    return mixture


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
