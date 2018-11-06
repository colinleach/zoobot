# from scipy import stats
import numpy as np
import matplotlib
matplotlib.use('Agg') # TODO move this to .matplotlibrc

from zoobot.estimators import make_predictions
from zoobot.uncertainty import sample_statistics


def verify_uncertainty(model, subjects, true_params, n_samples):
    # TODO not qiute working yet - coverage not quite coming out right!
    results = make_predictions.get_samples_of_subjects(model, subjects, n_samples)
    # do a KDE for each subject to get posterior function
    # how many subjects were within each confidence interval?
    # posteriors = [samples_to_posterior(results[n, :]) for n in range(len(results))]
    # sig1_alpha = 0.32
    sig2_alpha = 0.05
    # sig3_alpha = 0.003
    sig2_intervals = np.array([sample_statistics.samples_to_interval(results[n], alpha=sig2_alpha) for n in range(len(results))])
    within_interval = (true_params > sig2_intervals[:, 0]) & (true_params < sig2_intervals[:, 1])
    print(np.sum(within_interval))
    return float(np.sum(within_interval)) / float(len(subjects))  # coverage fraction
