
import pandas as pd
import numpy as np
import sklearn.linear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns


def evaluate_discrete_coverage(volunteer_votes, sample_probs_by_k):
    data = []
    n_subjects = sample_probs_by_k.shape[0]
    n_samples = sample_probs_by_k.shape[1]
    max_possible_k = sample_probs_by_k.shape[2]
    for subject_n in range(n_subjects):
        for sample_n in range(n_samples):
            for max_error_in_k in range(max_possible_k + 1):  # include max_error = max_k in range
                most_likely_k = sample_probs_by_k[subject_n, sample_n, :].argmax()
                max_k = np.min([most_likely_k + max_error_in_k, max_possible_k])
                min_k = np.max([most_likely_k - max_error_in_k, 0])
                prediction = sample_probs_by_k[subject_n, sample_n, min_k:max_k+1].sum()  # include max_k in slice
                observed = float(min_k <= volunteer_votes[subject_n] <= max_k)
                data.append({
                        'max_state_error': max_error_in_k,
                        'prediction': prediction,
                        'observed': observed
                    })
    df = pd.DataFrame(data=data)
    return df


def reduce_coverage_df(df):
    return df.groupby('max_state_error').agg({'prediction': 'mean', 'observed': 'mean'}).reset_index()


def calibrate_predictions(df):
    assert len(df) >= 4
    lr = sklearn.linear_model.LogisticRegression()
    df = df.sample(frac=1).reset_index(drop=True)
    train_df, test_df = df[:int(len(df)/4)], df[int(len(df)/4):]
    X_train = np.array(train_df['prediction']).reshape(-1, 1)
    X_test = np.array(test_df['prediction']).reshape(-1, 1)
    y_train = np.array(train_df['observed'])                
    lr.fit(X_train, y_train)
    test_df['prediction_calibrated'] = lr.predict_proba(X_test)[:,1]
    # print('Calibrated tidal predictions: {}. Non-tidal: {}'.format(np.sum(test_df['prediction'] > 0.5), np.sum(test_df['prediction'] < 0.5)))
    # print('Calibrated tidal truth: {}. Non-tidal: {}'.format(np.sum(test_df['true_label'] > 0.5), np.sum(test_df['true_label'] < 0.5)))
    return test_df



def plot_coverage_df(df, save_loc):
    fig, ax = plt.subplots()
    cols_to_plot = ['prediction', 'observed']
    if 'prediction_calibrated' in df.columns:
        cols_to_plot.append('prediction_calibrated')
    for col in cols_to_plot:
        sns.lineplot(data=df, x='max_state_error', y=col, ax=ax)
    ax.legend(cols_to_plot)
    ax.set_xlabel('Max error in states')
    ax.set_ylabel('Probability or Frequency')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))  # must expect 'x' kw arg
    fig.tight_layout()
    fig.savefig(save_loc)
    # TODO axis formatter for ints only
