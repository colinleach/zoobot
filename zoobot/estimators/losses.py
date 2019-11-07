import numpy as np
import tensorflow as tf


def calculate_binomial_loss(labels, predictions):
    scalar_predictions = get_scalar_prediction(predictions)  # softmax, get the 2nd neuron
    return binomial_loss(labels, scalar_predictions)


def get_scalar_prediction(prediction):
    return tf.nn.softmax(prediction)[:, 1]


# requires that labels be continguous by question - easily satisfied
def get_schema_from_label_cols(label_cols, questions):
    print('label_cols: {}'.format(label_cols))

    # current
    schema = []
    # if 'smooth' in questions:
    schema.append([
            label_cols.index('smooth-or-featured_smooth'),
            label_cols.index('smooth-or-featured_featured-or-disk')
            # TODO add artifact?
    ])
    # if 'spiral' in questions:
    schema.append([
            label_cols.index('has-spiral-arms_yes'),
            label_cols.index('has-spiral-arms_no')
    ])
    print('schema: {}'.format(schema))
    return schema  # do not convert to tensor, use tf.function() to trace np and make many graphs


def get_indices_from_label_cols(label_cols, questions):
    """
    Get indices for use with tf.dynamic_slice
    Example use:

    questions = ['q1', 'q2']
    label_cols = ['q1_a1', 'q1_a2', 'q2_a1', 'q2_a2']

    Returns:
    indices = [0, 0, 1, 1]
    """
    indices = np.zeros(len(label_cols))
    for question_n, question in enumerate(questions):
        for column_n, label_col in enumerate(label_cols):
            if label_col.startswith(question):
                indices[column_n] = question_n
    return tf.constant(indices.astype(int), dtype=tf.int32)


@tf.function
def multiquestion_loss(labels, predictions, question_index_groups, num_questions):
    # very important that question_index_groups is fixed, else tf autograph will mess up this for loop
    # answer_slices = question_index_groups.items()  # list of list of indices e.g. [[0, 1], [3, 4]]
    # all_losses = tf.map_fn(
    #     lambda x: multinomial_loss(labels[:, x[0]:x[1]], predictions[:, x[0]:x[1]]),
    #     question_index_groups
    # )
    # smooth_loss = multinomial_loss(labels[:, :2], predictions[:, :2])
    # tf.summary.histogram('smooth_loss', smooth_loss)
    # spiral_loss = multinomial_loss(labels[:, 2:], predictions[:, 2:])
    # tf.summary.histogram('spiral_loss', spiral_loss)
    # # TODO good view into each loss
    # total_loss = tf.reduce_mean(smooth_loss + spiral_loss)
    # tf.summary.histogram('total_loss', total_loss)
    # return total_loss

    q_losses = []
    for q_n in range(len(question_index_groups)):
        q_indices = question_index_groups[q_n]
        q_start = q_indices[0]
        q_end = q_indices[1]
        print('q_start', q_start)
        print('q_end', q_end)
        q_losses.append(multinomial_loss(labels[:, q_start:q_end+1], predictions[:, q_start:q_end+1]))
        # tf.summary.histogram('smooth_loss', q_loss)
        total_loss = tf.stack(q_losses, axis=1)

    # TODO good view into each loss
    tf.summary.histogram('total_loss', total_loss)
    return total_loss  # leave the reduce_sum to the estimator

    # do next
    # not really sure why tf separately requires num_partitions to be specified...?
    # labels_dim = tf.shape(input=labels)[1]
    # batch_dim = tf.shape(input=labels)[0]
    # copied_indices = tf.tile(question_index_groups, tf.expand_dims(batch_dim, axis=0))
    # tiled_indices = tf.reshape(copied_indices, (batch_dim, labels_dim))
    # labels_by_q = tf.dynamic_partition(labels, tiled_indices, num_partitions=num_questions)
    # predictions_by_q = tf.dynamic_partition(predictions, tiled_indices, num_partitions=num_questions)

    # print_op = tf.print('copied_indices', copied_indices, 'tiled_indices', tiled_indices, 'labels_by_q', labels_by_q,  'predictions_by_q', predictions_by_q)
    # with tf.control_dependencies([print_op]):
    #     copied_indices = tf.identity(copied_indices)
    # return copied_
    # losses_list = [multinomial_loss(question_labels, question_predictions) for question_labels, question_predictions in zip(labels_by_q, predictions_by_q)]
    # losses = tf.concat(losses, axis=1)
    # return losses  # reduce later!


def multinomial_loss(successes, expected_probs, output_dim=2):
    # for this to be correct, predictions must sum to 1 and successes must sum to n_trials
    # negative log loss, of course
    # successes x, probs p: tf.sum(x*log(p)). Each vector x, p of length k.

    loss = -tf.reduce_sum(input_tensor=successes * tf.math.log(expected_probs + tf.constant(1e-8, dtype=tf.float32)), axis=1)
    for n in range(output_dim):  # careful, output_dim must be fixed
        tf.compat.v1.summary.histogram('successes_{}'.format(n), successes[:, n])
        tf.compat.v1.summary.histogram('expected_probs_{}', expected_probs[:, n])
    tf.compat.v1.summary.histogram('loss', loss)
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
