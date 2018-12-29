
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_discrete_coverage(volunteer_votes, sample_probs_by_k):
    data = []
    for max_error_in_k in range(20):
        for subject_n in range(sample_probs_by_k.shape[0]):
            for sample_n in range(sample_probs_by_k.shape[1]):
                most_likely_k = sample_probs_by_k[subject_n, sample_n, :].argmax()
                max_k = np.min([most_likely_k + max_error_in_k, 40])
                min_k = np.max([most_likely_k - max_error_in_k, 0])
                prediction = sample_probs_by_k[subject_n, sample_n, min_k:max_k+1].sum()
                observed = float(min_k <= volunteer_votes[subject_n] <= max_k)
                data.append(
                    {
                        'max_state_error': max_error_in_k,
                        'probability': prediction,
                        'observed': False
                    })
                data.append(
                    {
                        'max_state_error': max_error_in_k,
                        'probability': observed,
                        'observed': True
                    }
                )
    df = pd.DataFrame(data=data)
    return df


def plot_coverage_df(df, save_loc):
    sns.lineplot(data=df, x='max_state_error', y='probability', hue='observed')
    plt.xlabel('Max error in states')
    plt.ylabel('Probability or Frequency')
    plt.tight_layout()
    plt.savefig(save_loc)
    # TODO axis formatter for ints only


if __name__ == '__main__':
    samples_loc = '/home/mike/repos/zoobot/analysis/uncertainty/al-binomial/five_conv_fractions/results.txt'
    samples = np.loadtxt(samples_loc)
    from zoobot.estimators import make_predictions
    bin_probs = make_predictions.bin_prob_of_samples(samples, n_draws=40)
    # TODO volunteer votes - labels are not actually saved!
    coverage_df = evaluate_discrete_coverage(volunteer_votes, bin_probs)
