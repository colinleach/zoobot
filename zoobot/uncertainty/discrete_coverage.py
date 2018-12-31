
import pandas as pd
import numpy as np
import sklearn
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


def reduce_coverage_df(df):
    predicted_df = df[~df['observed']]
    observed_df = df[df['observed']]
    assert len(predicted_df) == len(observed_df)

    predicted_grouped = predicted_df.groupby('max_state_error').agg({'probability': 'mean'}).reset_index().set_index('max_state_error')
    observed_grouped = observed_df.groupby('max_state_error').agg({'probability': 'mean'}).reset_index().set_index('max_state_error')

    # rename columns
    predicted_grouped['prediction'] = predicted_grouped['probability']
    del predicted_grouped['probability']
    observed_grouped['frequency'] = observed_grouped['probability']
    del observed_grouped['probability']

    # horizontal concat on max_state_error index
    return pd.concat([predicted_grouped, observed_grouped], axis=1).reset_index()


def calibrate_predictions(df):
    # TODO using reduced df
    lr = sklearn.linear_model.LogisticRegression()
    df = df.sample(frac=1).reset_index(drop=True)
    train_df, test_df = df[:int(len(df)/4)], df[int(len(df)/4):]
    X_train = np.array(train_df['prediction']).reshape(-1, 1)
    X_test = np.array(test_df['prediction']).reshape(-1, 1)
    y_train = np.array(train_df['frequency'])                
    lr.fit(X_train, y_train)
    test_df['prediction_calibrated'] = lr.predict_proba(X_test)[:,1]
    # print('Calibrated tidal predictions: {}. Non-tidal: {}'.format(np.sum(test_df['prediction'] > 0.5), np.sum(test_df['prediction'] < 0.5)))
    # print('Calibrated tidal truth: {}. Non-tidal: {}'.format(np.sum(test_df['true_label'] > 0.5), np.sum(test_df['true_label'] < 0.5)))
    return test_df



def plot_coverage_df(df, save_loc):
    sns.lineplot(data=df, x='max_state_error', y='probability', hue='observed')
    plt.xlabel('Max error in states')
    plt.ylabel('Probability or Frequency')
    plt.tight_layout()
    plt.savefig(save_loc)
    # TODO axis formatter for ints only
