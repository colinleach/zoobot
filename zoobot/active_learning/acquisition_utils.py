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

# from shared_astro_utils import plotting_utils
from zoobot.estimators import make_predictions, dirichlet_stats


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
    joint_samples = np.concatenate(samples_list, axis=2)  # concat dropout dimension
    assert joint_samples.shape[1] == len(schema.answers)
    prob_of_answers = dirichlet_stats.dirichlet_prob_of_answers(joint_samples, schema)

    logging.warning('Only optimising over four questions')
    for q in schema.questions:
        logging.warning('Only optimising smooth/featured!')
        # if q.text in ['smooth-or-featured', 'has-spiral-arms', 'bar', 'bulge-size']:
        if q.text in ['smooth-or-featured']:
            logging.info(q.text)
            # expected_votes_list = [get_expected_votes_ml(samples, q, retirement, schema, round=True) for samples in samples_list]
            # print(expected_votes_list)

            samples_list_by_q = [samples[:, q.start_index:q.end_index+1] for samples in samples_list]  # shape (model, galaxy, answer, dropout)

            # print('Calculating predictive entropy')
            # predictive_entropy = dirichlet_predictive_entropy(samples_list_by_q, expected_votes_list)
            # print('Calculating expected entropy')
            # expected_entropy = dirichlet_expected_entropy(samples_list_by_q, expected_votes_list)

            logging.info('Calculating predictive entropy')
            predictive_entropy = dirichlet_predictive_entropy_alpha(samples_list_by_q)
            # print(predictive_entropy.shape)

            logging.info('Calculating expected entropy')
            expected_entropy = dirichlet_expected_entropy_alpha(samples_list_by_q)
            # print(expected_entropy.shape)
            assert predictive_entropy.shape == expected_entropy.shape
            mi_for_q = predictive_entropy - expected_entropy

            logging.info('Calculating joint p of being asked')
            prev_q = q.asked_after
            if prev_q is None:
                joint_p_of_asked = 1.
            else:
                joint_p_of_asked = schema.joint_p(prob_of_answers, q.asked_after.text)

            all_predictive_entropy.append(predictive_entropy)
            all_expected_entropy.append(expected_entropy)
            all_expected_mi.append(mi_for_q * joint_p_of_asked)
        else:
            logging.warning('Skipping {}'.format(q.text))

    logging.info('Acquisition calculations complete')
    return np.array(all_expected_mi).transpose(), np.array(all_predictive_entropy).transpose(), np.array(all_expected_entropy).transpose()


def get_expected_votes_ml(samples, q, retirement, schema, round_votes):
    prob_of_answers = dirichlet_stats.dirichlet_prob_of_answers(samples, schema)  # mean over both models. Prob given q is asked!
    prev_q = q.asked_after
    if prev_q is None:
        expected_votes = tf.ones(len(samples)) * retirement
    else:
        joint_p_of_asked = schema.joint_p(prob_of_answers, prev_q.text)  # prob of getting the answer needed to ask this question
        expected_votes = joint_p_of_asked * retirement
    if round_votes:
        return tf.round(expected_votes)
    else:
        return expected_votes


def get_expected_votes_human(label_df, q, retirement, schema, round_votes):
    prob_of_answers = label_df[[a + '_fraction' for a in schema.label_cols]].values
    prev_q = q.asked_after
    if prev_q is None:
        expected_votes = tf.ones(len(label_df)) * retirement
    else:
        joint_p_of_asked = schema.joint_p(prob_of_answers, prev_q.text)  # prob of getting the answer needed to ask this question
        expected_votes = joint_p_of_asked * retirement
    if round_votes:
        return tf.round(expected_votes)
    else:
        return expected_votes


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


# entropy in *votes*

def dirichlet_expected_entropy_votes(list_of_samples_for_q, list_of_expected_votes):
    expected_entropies = []
    for n, samples_for_q in enumerate(list_of_samples_for_q):  # for each model
        expected_votes = list_of_expected_votes[n]
        # predictive entropy per model is an expected entropy for one model (now over dropout)
        expected_entropies.append(dirichlet_predictive_entropy_per_model(samples_for_q, expected_votes))
    return np.mean(expected_entropies, axis=0)


def dirichlet_predictive_entropy_votes(list_of_samples_for_q, list_of_expected_votes):
    # print(list_of_samples_for_q[0].shape)
    joint_samples = np.concatenate(list_of_samples_for_q, axis=2)  # concat along dropout (sample) axis
    stacked_expected_votes = np.stack(list_of_expected_votes, axis=1)
    # print(stacked_expected_votes)
    joint_expected_votes = np.round(np.mean(stacked_expected_votes, axis=1))  # take the average expected votes. But won't be int?
    # now we consider these samples from one model
    # print(joint_expected_votes)
    return dirichlet_predictive_entropy_per_model(joint_samples, joint_expected_votes)


# entropy in *alpha*

def dirichlet_expected_entropy_alpha(list_of_samples_for_q):
    mixture_for_model = [dirichlet_stats.DirichletEqualMixture(galaxy_answer_dropout) for galaxy_answer_dropout in list_of_samples_for_q]
    entropies_by_model = [mixture.entropy_estimate() for mixture in mixture_for_model]
    return np.mean(entropies_by_model, axis=0)  # average model dimension

def dirichlet_predictive_entropy_alpha(list_of_samples_for_q):
    joint_samples = np.concatenate(list_of_samples_for_q, axis=2)  # concat along dropout (sample) axis, making (galaxy, answer, dropout_both) shape
    mixture = dirichlet_stats.DirichletEqualMixture(joint_samples)  # mixture has batch (galaxy) dimension
    entropies_by_galaxy = mixture.entropy_estimate()
    return np.array(entropies_by_galaxy)


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
    # this is very slow, but permutations_by_gaalxy can only work on single galaxies as num. of permutations changes
    for galaxy_n, galaxy in enumerate(samples_for_q):  # includes dropout
        galaxy_with_dummy_batch = np.expand_dims(galaxy, axis=0)  # mixture requires batch dimension
        # print(galaxy_n, expected_votes[galaxy_n], n_answers, n_samples)
        # older slower version
        # mixture = dirichlet_mixture(galaxy_with_dummy_batch, expected_votes[galaxy_n], n_samples)

        mixture = dirichlet_stats.DirichletEqualMixture(expected_votes[galaxy_n], galaxy_with_dummy_batch)

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
    # .e.g [[0, 3], [1, 2], ... [3, 0]] for n=3, k=2


def dirichlet_entropy_in_concentrations(samples_for_q, expected_votes):
    # samples should have shape (batch, answers_for_q), with values being the concentration of each galaxy/answer
    dist = tfp.distributions.DirichletMultinomial(expected_votes, samples_for_q, validate_args=True)
    return dist.entropy


# not currently used
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

# not currently used
# def save_acquisition_examples(images, acq_values, acq_string, save_dir):
#     """[summary]
    
#     Args:
#         images (np.array): of form [n_subjects, height, width, channels]. NOT a list.
#         acq_values ([type]): [description]
#         acq_string ([type]): [description]
#         save_dir ([type]): [description]
#     """
#     assert isinstance(images, np.ndarray)
#     assert isinstance(acq_values, np.ndarray)
#     # show galaxies with max/min variance, or top/bottom 20% of variance (more representative)
#     sorted_galaxies = images[acq_values.argsort()]
#     min_gals = sorted_galaxies
#     max_gals = sorted_galaxies[::-1]  # reverse
#     low_galaxies = sorted_galaxies[:int(len(images)/5.)]
#     high_galaxies = sorted_galaxies[int(-len(images)/5.):]
#     np.random.shuffle(low_galaxies)   # inplace
#     np.random.shuffle(high_galaxies)  # inplace

#     galaxies_to_show = [
#         {
#             'galaxies': min_gals, 
#             'save_loc': os.path.join(save_dir, 'min_{}.png'.format(acq_string))
#         },
#         {
#             'galaxies': max_gals,
#             'save_loc': os.path.join(save_dir, 'max_{}.png'.format(acq_string))
#         },
#         {
#             'galaxies': high_galaxies,
#             'save_loc': os.path.join(save_dir, 'high_{}.png'.format(acq_string))
#         },
#         {
#             'galaxies': low_galaxies,
#             'save_loc': os.path.join(save_dir, 'low_{}.png'.format(acq_string))
#         },
#     ]

#     # save images
#     for galaxy_set in galaxies_to_show:
#         assert len(galaxy_set['galaxies']) != 0
#         plotting_utils.plot_galaxy_grid(
#             galaxies=galaxy_set['galaxies'],
#             rows=9,
#             columns=3,
#             save_loc=galaxy_set['save_loc']
#         )
