import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from zoobot.active_learning import acquisition_utils

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


def get_posteriors(samples, catalog, id_strs, question, answer):
    all_galaxy_posteriors = []
    for sample_n, sample in enumerate(samples):
        galaxy = catalog[catalog['id_str'] == id_strs[sample_n]].squeeze()
        galaxy_posteriors = get_galaxy_posteriors(sample, galaxy, question, answer)
        all_galaxy_posteriors.append(galaxy_posteriors)
    return all_galaxy_posteriors

def get_galaxy_posteriors(sample, galaxy, question, answer):
    n_samples = sample.shape[-1]
    cols = [a.text for a in question.answers]
    total_votes = galaxy[cols].sum().astype(np.float32)
#     total_votes = galaxy[question.text + '_total-votes'].astype(np.float32)

    votes = np.arange(0., total_votes+1)
    x = np.stack([votes, total_votes-votes], axis=-1)  # also need the counts for other answer, no
    votes_this_answer = x[:, answer.index - question.start_index]
    
    all_probs = []
    for d in range(n_samples):
        concentrations = tf.constant(sample[question.start_index:question.end_index+1, d].astype(np.float32))
        probs = tfp.distributions.DirichletMultinomial(total_votes, concentrations).prob(x)
        all_probs.append(probs)
        
    return votes_this_answer, np.array(all_probs)



# only used for posthoc evaluation, not when training

def dropout_loss_multiq(labels, predictions, question_index_groups):  # pasted
    q_losses = []
    for q_n in range(len(question_index_groups)):
        q_indices = question_index_groups[q_n]
        q_start = q_indices[0]
        q_end = q_indices[1]
        q_loss = dropout_loss(labels[:, q_start:q_end+1], predictions[:, q_start:q_end+1])
        q_losses.append(q_loss)
    
    total_loss = tf.stack(q_losses, axis=1)
    return total_loss  # leave the reduce_sum to the estimator

def dropout_loss(labels_q, predictions_q):
    n_samples = predictions_q.shape[-1]
    total_votes = labels_q.sum(axis=1).squeeze()
    log_probs = []
#     print(predictions_q.shape,total_votes.shape, n_samples)
    for n, galaxy in enumerate(predictions_q):
        mixture = acquisition_utils.dirichlet_mixture(np.expand_dims(galaxy, axis=0), total_votes[n], n_samples)
        log_probs.append(mixture.log_prob(labels_q[n]))
    return -np.squeeze(np.array(log_probs))  # negative log prob