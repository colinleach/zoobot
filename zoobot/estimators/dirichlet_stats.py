import json

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from zoobot.active_learning import acquisition_utils
from zoobot.estimators import mixture_stats

class EqualMixture():

    def __init__(self):
        raise NotImplementedError

    @property
    def batch_shape(self):
        return self.distributions[0].batch_shape

    @property
    def event_shape(self):
        return self.distributions[0].event_shape
    
    def log_prob(self, x):
        log_probs_by_dist = [dist.log_prob(x) for dist in self.distributions]
        return tf.transpose(log_probs_by_dist)
    
    def prob(self, x):  # just the same, but prob not log prob
        return tf.math.exp(self.log_prob(x))

    def mean_prob(self, x):
        prob = self.prob(x)
        return tf.reduce_mean(prob, axis=-1)

    def mean_log_prob(self, x):
        return tf.math.log(self.mean_prob(x))

    def mean(self):  # i.e. expected 
        return tf.reduce_mean([dist.mean() for dist in self.distributions], axis=0)


class DirichletEqualMixture(EqualMixture):

    def __init__(self, concentrations):
        # self.concentrations = tf.constant(concentrations, dtype=tf.float32)
        self.concentrations = concentrations.astype(np.float32)
        self.n_distributions = self.concentrations.shape[2]
        self.distributions = [
            tfp.distributions.Dirichlet(concentrations[:, :, n], validate_args=True)
            for n in range(self.n_distributions)
        ]

    def entropy_estimate(self):
        upper = self.entropy_upper_bound()
        lower = self.entropy_lower_bound()
        return lower + (upper-lower)/2.  # midpoint between bounds

    def entropy_upper_bound(self):
        return np.array([mixture_stats.entropy_upper_bound(galaxy_conc, weights=np.ones(self.n_distributions)/self.n_distributions) for galaxy_conc in self.concentrations])

    def entropy_lower_bound(self):
        return np.array([mixture_stats.entropy_lower_bound(galaxy_conc, weights=np.ones(self.n_distributions)/self.n_distributions) for galaxy_conc in self.concentrations])

class DirichletMultinomialEqualMixture(EqualMixture):

    def __init__(self, total_votes, *args, **kwargs):
        """
        To use same interface as tfp, but much faster for equal mixtures of dirichlet-multinomials

        Args:
            total_votes ([type]): of shape (batch)
            concentrations ([type]): of shape (batch, answer, repetition), for one question only
        """
        
        self.total_votes = np.array(total_votes).astype(np.float32)
        # otherwise the same as DirichletEqualMixture
        DirichletEqualMixture.__init__(self, *args, **kwargs)


def get_hpd(x, p, ci=0.8):
    assert np.isclose(p.sum(), 1, atol=0.001)
    # here, x is discrete posterior p's, not samples as with agnfinder
    mode_index = np.argmax(p)
    lower_index = mode_index
    higher_index = mode_index
    while True:
        confidence = p[lower_index:higher_index+1].sum()
#         print(lower_index, higher_index, confidence)
        if confidence >= ci:
            break  # discrete so will be a little under or a little over
        else:
            lower_index = max(0, lower_index-1)
            higher_index = min(len(x)-1, higher_index+1)

    # these indices will give a symmetric interval of at least ci, but not exactly - and will generally overestimate


    return (x[lower_index], x[higher_index]), confidence
        

def get_coverage(posteriors, true_values):
    results = []
    for ci_width in np.linspace(0.1, 0.95):
        for target_n, (x, posterior) in enumerate(posteriors):
            true_value = true_values[target_n]
            credible_interval_width = 0.8
#             modes = get_hpd(posterior, ci=ci_width)
#             within_any_ci = any([x[0] < true_value < x[1] for x in modes])
            (lower_lim, higher_lim), confidence = get_hpd(x, posterior, ci=ci_width)
            within_any_ci = lower_lim <= true_value <= higher_lim
            results.append({
                'ci_width': ci_width,
                'confidence': confidence,
#                 'hpd_min': hpd[0],
#                 'hpd_max': hpd[1],
                'true_value': true_value,
                'true_within_hpd': within_any_ci
            })
    return pd.DataFrame(results)


def get_true_values(catalog, id_strs, answer):
    true_values = []
    for id_str in id_strs:
        galaxy = catalog[catalog['id_str'] == id_str].squeeze()
        true_values.append(galaxy[answer.text])
    return true_values

# samples and catalog are aligned - so let's just use them as one dataframe instead of two args
def get_posteriors(samples, catalog, id_strs, question, answer):
    """[summary]

    Args:
        samples ([type]): [description]
        catalog ([type]): [description]
        id_strs ([type]): [description]
        question ([type]): [description]
        answer ([type]): [description]

    Returns:
        list: posteriors like [yes_votes_arr, p_of_each] for each galaxy
    """
    all_galaxy_posteriors = []
    for sample_n, sample in enumerate(samples):
        galaxy = catalog[catalog['id_str'] == id_strs[sample_n]].squeeze()
        galaxy_posteriors = get_galaxy_posteriors(sample, galaxy, question, answer)
        all_galaxy_posteriors.append(galaxy_posteriors)
    return all_galaxy_posteriors


def get_galaxy_posteriors(sample, galaxy, question, answer):
    n_samples = sample.shape[-1]
    cols = [a.text for a in question.answers]
    assert len(cols) == 2 # Binary only!
    total_votes = galaxy[cols].sum().astype(np.float32)

    votes = np.arange(0., total_votes+1)
    x = np.stack([votes, total_votes-votes], axis=-1)  # also need the counts for other answer, no. 
    votes_this_answer = x[:, answer.index - question.start_index]
    
    # could refactor with new equal mixture class
    all_probs = []
    for d in range(n_samples):
        concentrations = tf.constant(sample[question.start_index:question.end_index+1, d].astype(np.float32))
        probs = tfp.distributions.DirichletMultinomial(total_votes, concentrations).prob(x)
        all_probs.append(probs)
        
    return votes_this_answer, np.array(all_probs)


def load_all_concentrations(df, concentration_cols):
    temp = []
    for col in concentration_cols:
        temp.append(np.stack(df[col].apply(json.loads).values, axis=0))
    return np.stack(temp, axis=2).transpose(0, 2, 1)


def dirichlet_prob_of_answers(samples, schema):
    # mean probability (including dropout) of an answer being given. 
    # samples has (batch, answer, dropout) shape
    p_of_answers = []
    for q in schema.questions:
        samples_by_q = samples[:, q.start_index:q.end_index+1, :]
        p_of_answers.append(DirichletMultinomialEqualMixture(total_votes=1, concentrations=samples_by_q).mean().numpy())

    p_of_answers = np.concatenate(p_of_answers, axis=1)
    return p_of_answers



# only used for posthoc evaluation, not when training
def dirichlet_mixture_loss(labels, predictions, question_index_groups):  # pasted
    q_losses = []
    for q_n in range(len(question_index_groups)):
        q_indices = question_index_groups[q_n]
        q_start = q_indices[0]
        q_end = q_indices[1]
        q_loss = dirichlet_mixture_loss_per_question(labels[:, q_start:q_end+1], predictions[:, q_start:q_end+1])
        q_losses.append(q_loss)
    
    total_loss = tf.stack(q_losses, axis=1)
    return total_loss  # leave the reduce_sum to the estimator

# this works but is very slow
# def dirichlet_mixture_loss_per_question(labels_q, predictions_q):
#     n_samples = predictions_q.shape[-1]
#     total_votes = labels_q.sum(axis=1).squeeze()
#     log_probs = []
# #     print(predictions_q.shape,total_votes.shape, n_samples)
#     for n, galaxy in enumerate(predictions_q):
#         mixture = acquisition_utils.dirichlet_mixture(np.expand_dims(galaxy, axis=0), total_votes[n], n_samples)
#         log_probs.append(mixture.log_prob(labels_q[n]))
#     return -np.squeeze(np.array(log_probs))  # negative log prob

def dirichlet_mixture_loss_per_question(labels_q, concentrations_q):
    n_samples = concentrations_q.shape[-1]
    total_votes = labels_q.sum(axis=1).squeeze()
    mean_log_probs = DirichletMultinomialEqualMixture(total_votes, concentrations_q).mean_log_prob(labels_q) 
    return -np.squeeze(np.array(mean_log_probs))  # negative log prob


# deprecated
# def dirichlet_mixture(samples_for_q, expected_votes, n_samples):
#     """[summary]

#     Args:
#         samples_for_q ([type]): shape (batch, answer, dropout)
#         expected_votes ([type]): scalar, I am doing something wrong as it doesn't seem to like a batch distribution for n despite the example
#         n_samples ([type]): scalar

#     Returns:
#         [type]: [description]
#     """
#     assert samples_for_q.ndim == 3  # must have galaxies dimension (:1 will work)
#     # samples_for_q has 

#     component_probs = tf.zeros(n_samples) / n_samples
#     categorical = tfp.distributions.Categorical(logits=component_probs, validate_args=True)

#     # move dropout to (leading) batch dimensions, concentrations to (last) event dim. Now (batch, dropout, concentrations)
#     concentrations = tf.transpose(samples_for_q, (0, 2, 1))
#     # print(concentrations.shape)
#     component_distribution = tfp.distributions.DirichletMultinomial(expected_votes, concentrations, validate_args=True)
#     # print(component_distribution)

#     mixture = tfp.distributions.MixtureSameFamily(
#         categorical,
#         component_distribution,
#         validate_args=True
#     )
#     return mixture


# deprecated
# def get_kde(x):
#     estimator = sm.nonparametric.KDEUnivariate(x)
#     estimator.fit(fft=True)
#     return estimator



# temporary kde mutual info func, requires genuinely several models
# def multimodel_bald(model_predictions, min_entropy=-5.5, max_entropy=10):
#     individual_entropies = np.array([get_kde(predictions).entropy for predictions in model_predictions])
#     mean_individual_entropy = individual_entropies.mean()
#     expected_entropy = get_kde(model_predictions.flatten()).entropy
#     if individual_entropies.min() < min_entropy:  # occasionally, very nearby predictions can cause insanely spiked kdes
#         return np.nan
#     elif individual_entropies.max() > max_entropy:
#         return np.nan
#     else:
#         return expected_entropy - mean_individual_entropy

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


# def dirichlet_mutual_information(samples_for_q, expected_votes):
#     # samples_for_q has shape (batch, concentration, dropout)
#     # MI = entropy over all samples - mean entropy of each sample

#     # predictive_entropy = dirichlet_mixture(samples_for_q, expected_votes).entropy.numpy()  # or .entropy_lower_bound
#     predictive_entropy = dirichlet_predictive_entropy(samples_for_q, expected_votes)
#     expected_entropy = dirichlet_expected_entropy(samples_for_q, expected_votes)
#     return predictive_entropy - expected_entropy  # (batch) shape


