import logging
from typing import List

import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp


class Question():

    def __init__(self, text:str, label_cols: List, version: str):
        self.text = text

        if version == 'decals':
            self.answers = create_answers_decals(self, label_cols)
        elif version == 'gz2':
            self.answers = create_answers_gz2(self, label_cols)
        else:
            raise ValueError(f'Version {version} not recognised')

        self.start_index = min(a.index for a in self.answers)
        self.end_index = max(a.index for a in self.answers)
        assert [self.start_index <= a.index <= self.end_index for a in self.answers]

        self._asked_after = None

    @property
    def asked_after(self):
        return self._asked_after

    def __repr__(self):
        return f'{self.text}, indices {self.start_index} to {self.end_index}, asked after {self.asked_after}'

class Answer():

    def __init__(self, text, question, index):
        self.text = text
        self.question = question

        self.index = index
        self._next_question = None

    @property
    def next_question(self):
        return self._next_question


def create_answers_decals(question, label_cols):
    question_text = question.text
    if question_text == 'smooth-or-featured':
        answer_substrings = ['_smooth', '_featured-or-disk']
    elif question_text == 'disk-edge-on':
        answer_substrings = ['_yes', '_no']
    elif question_text == 'has-spiral-arms':
        answer_substrings = ['_yes', '_no']
    elif question_text == 'spiral-winding':
        answer_substrings = ['_tight', '_medium', '_loose']
    elif question_text == 'bar':
        answer_substrings = ['_strong', '_weak', '_no']
    elif question_text == 'bulge-size':
        answer_substrings = ['_dominant', '_large', '_moderate', '_small', '_none']
    elif question_text == 'something-odd':
        answer_substrings = ['_yes', '_no']
    elif question_text == 'how-rounded':
        answer_substrings = ['_round', '_in-between', '_cigar-shaped']
    elif question_text == 'edge-on-bulge':
        answer_substrings = ['_boxy', '_none', '_rounded']
    elif question_text == 'spiral-arm-count':
        answer_substrings = ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell']
    elif question_text == 'merging':
        answer_substrings = ['_none', '_minor-disturbance', '_major-disturbance', '_merger']
    else:
        print(question_text)
        raise ValueError(question.text + ' not recognised')
    return [Answer(question_text + substring, question, label_cols.index(question_text + substring)) for substring in answer_substrings]


def create_answers_gz2(question, label_cols):
    question_text = question.text
    if question_text == 'smooth-or-featured':
        answer_substrings = ['_smooth', '_featured-or-disk']
    elif question_text == 'disk-edge-on':
        answer_substrings = ['_yes', '_no']
    elif question_text == 'bar':
        answer_substrings = ['_yes', '_no']
    elif question_text == 'has-spiral-arms':
        answer_substrings = ['_yes', '_no']
    elif question_text == 'bulge-size':
        answer_substrings = ['_dominant', '_obvious', '_just-noticeable', '_no']
    elif question_text == 'something-odd':
        answer_substrings = ['_yes', '_no']
    elif question_text == 'how-rounded':
        answer_substrings = ['_round', '_in-between', '_cigar']
    elif question_text == 'bulge-shape':
        answer_substrings = ['_round', '_boxy', '_no-bulge']
    elif question_text == 'spiral-winding':
        answer_substrings = ['_tight', '_medium', '_loose']
    elif question_text == 'spiral-count':
        answer_substrings = ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell']
    else:
        print(question_text)
        raise ValueError(question.text + ' not recognised')
    return [Answer(question_text + substring, question, label_cols.index(question_text + substring)) for substring in answer_substrings]


def set_dependencies(questions):
    #  awkward to keep the partial version as it changes these dependencies
    if 'disk-edge-on' in [q.text for q in questions]:
        featured_branch_answer = 'disk-edge-on_no'
    else:
        featured_branch_answer = 'smooth-or-featured_featured-or-disk'  # skip it

    # luckily, these are the same in GZ2 and decals, just only some questions are asked
    dependencies = {
        'smooth-or-featured': None,  # always asked
        'disk-edge-on': 'smooth-or-featured_featured-or-disk',
        'has-spiral-arms': featured_branch_answer,
        'bar': featured_branch_answer,
        'bulge-size': featured_branch_answer,
        'something-odd': None,  # always asked
        'how-rounded': 'smooth-or-featured_smooth',
        'bulge-shape': 'disk-edge-on_yes',
        'spiral-winding': 'has-spiral-arms_yes',
        'spiral-count': 'has-spiral-arms_yes',
        'spiral-arm-count': 'has-spiral-arms_yes', # bad naming...
        'merging': None
    }

    for question in questions:
        prev_answer_text = dependencies[question.text]
        if prev_answer_text is not None:
            prev_answer = [a for q in questions for a in q.answers if a.text == prev_answer_text][0]  # will be exactly one match
            prev_answer._next_question = question
            question._asked_after = prev_answer
    # acts inplace

class Schema():
    """
Relate the df label columns tor question/answer groups and to tfrecod label indices
    """
    def __init__(self, label_cols: List, question_texts: List, version):
        """
        Requires that labels be continguous by question - easily satisfied
        
        Args:
            label_cols (List): columns (strings) which record k successes for each galaxy for each answer
            questions (List): semantic names of questions which group those answers
        """
        logging.info(f'Label cols: {label_cols} \n Questions: {question_texts}')
        self.label_cols = label_cols
        # self.questions = questions
        self.version = version

        """
        Be careful:
        - first entry should be the first answer to that question, by df column order
        - second entry should be the last answer to that question, similarly
        - answers in between will be included: these are used to slice
        - df columns must be contigious by question (e.g. not smooth_yes, bar_no, smooth_no) for this to work!
        """
        self.questions = [Question(question_text, label_cols, version=version) for question_text in question_texts]
        if len(self.questions) > 1:
            set_dependencies(self.questions)

        assert len(self.question_index_groups) > 0
        assert len(self.questions) == len(self.question_index_groups)

        print(self.named_index_groups)

    def get_answer(self, answer_text):
        try:
            return [a for q in self.questions for a in q.answers if a.text == answer_text][0]  # will be exactly one match
        except IndexError:
            raise ValueError('Answer not found: ', answer_text)

    def get_question(self, question_text):
        try:
            return [q for q in self.questions if q.text == question_text][0]  # will be exactly one match
        except  IndexError:
            raise ValueError('Question not found: ', question_text)
    
    @property
    def question_index_groups(self):
         # start and end indices of answers to each question in label_cols e.g. [[0, 1]. [1, 3]] 
        return [(q.start_index, q.end_index) for q in self.questions]


    @property
    def named_index_groups(self):
        return dict(zip(self.questions, self.question_index_groups))


    def joint_p(self, prob_of_answers, answer_text):
        assert prob_of_answers.ndim == 2  # batch, p. No 'per model', marginalise first
        # prob(answer) = p(that answer|that q asked) * p(that q_asked) i.e...
        # prob(answer) = p(that answer|that q asked) * p(answer before that q)
        answer = self.get_answer(answer_text)
        p_answer_given_question = prob_of_answers[:, answer.index]

        question = answer.question
        prev_answer = question.asked_after
        if prev_answer is None:
            return p_answer_given_question
        else:
            p_prev_answer = self.joint_p(prob_of_answers, prev_answer.text)  # recursive
            return p_answer_given_question * p_prev_answer

    @property
    def answers(self):
        answers = []
        for q in self.questions:
            for a in q.answers:
                answers.append(a)
        return answers

    # TODO write to disk




def calculate_binomial_loss(labels, predictions):
    scalar_predictions = get_scalar_prediction(predictions)  # softmax, get the 2nd neuron
    return binomial_loss(labels, scalar_predictions)


def get_scalar_prediction(prediction):
    return tf.nn.softmax(prediction)[:, 1]


    
def get_indices_from_label_cols(label_cols, questions):
    """
    Get indices for use with tf.dynamic_slice
    Example use:

    questions = ['q1', 'q2']
    label_cols = ['q1_a1', 'q1_a2', 'q2_a1', 'q2_a2']

    Returns:
    indices = [0, 0, 1, 1]
    """
    raise NotImplementedError('This has been deprecated, use get_schema above')
    # indices = np.zeros(len(label_cols))
    # for question_n, question in enumerate(questions):
    #     for column_n, label_col in enumerate(label_cols):
    #         if label_col.startswith(question):
    #             indices[column_n] = question_n
    # return tf.constant(indices.astype(int), dtype=tf.int32)


def dirichlet_loss(labels_for_q, concentrations_for_q):
    total_count = tf.reduce_sum(labels_for_q, axis=1)
    dist = tfp.distributions.DirichletMultinomial(total_count, concentrations_for_q)
    return -dist.log_prob(labels_for_q)  # important minus sign


def beta_loss(labels_for_q, alpha_for_q, beta_for_q, n_answers):
    # labels (batch, answer)
    # params (batch, param)
    # print(alpha_for_q.shape)
    # print(beta_for_q.shape)
    total_counts = tf.reduce_sum(labels_for_q, axis=1)

    if n_answers > 2:
        a_losses = []
        for answer_n in range(n_answers):
            labels_for_a = labels_for_q[:, answer_n]
            alpha_for_a = alpha_for_q[:, answer_n]
            beta_for_a = beta_for_q[:, answer_n]
            a_losses.append(beta_log_prob(labels_for_a, alpha_for_a, beta_for_a, total_counts))
        loss_by_question = -tf.reduce_sum(a_losses, axis=0)  # 0th axis is list index,1st is batch dim. Important minus!
    else:  # ignore the prediction for the other answer - one (alpha, beta) pair is enough. Loss only calculated for one answer (makes sense, I think)
        return -beta_log_prob(labels_for_q[:, 0], alpha_for_q[:, 0], beta_for_q[:, 0], total_counts)  # important minus!
    return loss_by_question

def beta_log_prob(successes, alpha, beta, total_counts):
    # print(total_counts.shape)  # (batch)
    # print(successes.shape)  # (batch)
    # print(alpha.shape)  # (batch)
    # print(beta.shape)  # (batch)
    # print('Alpha:', np.around(alpha.numpy(), 4))
    # print('Beta:', np.around(beta.numpy(), 4))
    # print('Successes: ', successes)
    # print('Total counts: ', total_counts)
    beta_bin_dist = tfp.distributions.BetaBinomial(total_counts, alpha, beta, validate_args=True)
    log_prob = beta_bin_dist.log_prob(successes)  # (batch) shape output
    # print('Log prob: ', np.around(log_prob.numpy(), 4))
    return log_prob


# def multiquestion_beta_loss(labels, predictions, question_index_groups):
#     q_losses = []
#     for q_n in range(len(question_index_groups)):
#         q_indices = question_index_groups[q_n]
#         q_start = q_indices[0]
#         q_end = q_indices[1]
#         total_count = labels[:, q_start:q_end+1]
#         q_
    
#     # TODO binary questions should either skip second loss output or not be calculated/enforced identical

# @tf.function
def multiquestion_loss(labels, predictions, question_index_groups):
    """[summary]
    
    Args:
        labels (tf.Tensor): (galaxy, k successes) where k successes dimension is indexed by question_index_groups
        predictions (tf.Tensor): coin-toss probabilities of success, matching shape of labels
        question_index_groups (list): Paired (tuple) integers of (first, last) indices of answers to each question, listed for all questions
    
    Returns:
        [type]: [description]
    """
    # very important that question_index_groups is fixed and discrete, else tf.function autograph will mess up 
    q_losses = []
    # print(predictions.shape)
    for q_n in range(len(question_index_groups)):
        q_indices = question_index_groups[q_n]
        q_start = q_indices[0]
        q_end = q_indices[1]
        # print(q_start, q_end)
        # q_loss = multinomial_loss(labels[:, q_start:q_end+1], predictions[:, q_start:q_end+1])
        # q_loss = beta_loss(
        #     labels_for_q=labels[:, q_start:q_end+1], 
        #     alpha_for_q=predictions[:, q_start:q_end+1, 0],
        #     beta_for_q=predictions[:, q_start:q_end+1, 1],
        #     n_answers=q_end+1-q_start
        # )
        q_loss = dirichlet_loss(labels[:, q_start:q_end+1], predictions[:, q_start:q_end+1])

        q_losses.append(q_loss)
    
    total_loss = tf.stack(q_losses, axis=1)
    return total_loss  # leave the reduce_sum to the estimator


def multinomial_loss(successes, expected_probs, output_dim=2):
    """
    For this to be correct, predictions must sum to 1 and successes must sum to n_trials (i.e. all answers to each question are known)
    Negative log loss, of course
    
    Args:
        successes (tf.Tensor): (galaxy, k_successes) where k_successes is indexed by each answer (e.g. [:, 0] = smooth votes, [:, 1] = featured votes)
        expected_probs (tf.Tensor): coin-toss probability of success, same dimensions as successes
        output_dim (int, optional): Number of answers (i.e. successes.shape[1]). Defaults to 2. TODO may remove?
    
    Returns:
        tf.Tensor: neg log likelihood of k_successes observed across all answers. With batch dimension.
    """
    # successes x, probs p: tf.sum(x*log(p)). Each vector x, p of length k.
    # important minus!
    loss = -tf.reduce_sum(input_tensor=successes * tf.math.log(expected_probs + tf.constant(1e-8, dtype=tf.float32)), axis=1)
    # print_op = tf.print('successes', successes, 'expected_probs', expected_probs)
    # with tf.control_dependencies([print_op]):
    return loss

def binomial_loss(labels, predictions):
    """Calculate likelihood of labels given predictions, if labels are binomially distributed
    
    Args:
        labels (tf.constant): of shape (batch_dim, 2) where 0th col is successes and 1st is total trials
        predictions (tf.constant): of shape (batch_dim) with values of prob. success
    
    Returns:
        (tf.constant): negative log likelihood of labels given prediction
    """
    one = tf.constant(1., dtype=tf.float32)
    # TODO may be able to use normal python types, not sure about speed
    epsilon = tf.constant(1e-8, dtype=tf.float32)

    # multiplication in tf requires floats
    successes = tf.cast(labels[:, 0], tf.float32)
    n_trials = tf.cast(labels[:, 1], tf.float32)
    p_yes = tf.identity(predictions)  # fail loudly if passed out-of-range values

    # negative log likelihood
    bin_loss = -( successes * tf.math.log(p_yes + epsilon) + (n_trials - successes) * tf.math.log(one - p_yes + epsilon) )
    tf.compat.v1.summary.histogram('bin_loss', bin_loss)
    tf.compat.v1.summary.histogram('bin_loss_clipped', tf.clip_by_value(bin_loss, 0., 50.))
    return bin_loss
