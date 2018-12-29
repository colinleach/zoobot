
import pandas as pd
import numpy as np


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