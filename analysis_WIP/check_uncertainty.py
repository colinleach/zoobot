import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics

from zoobot.estimators import make_predictions
from zoobot.tfrecord import read_tfrecord
from zoobot.uncertainty import dropout_calibration


if __name__ == '__main__':

    dropout = '05'

    # final model, dropout of 50%
    # predictor_loc = '/Data/repos/zoobot/runs/bayesian_panoptes_featured_si128_sf64_lfloat_regr/1541491411'

    predictor_loc = '/Data/repos/zoobot/runs/bayesian_panoptes_featured_si128_sf64_lfloat_regr/final_d{}'.format(dropout)

    model = make_predictions.load_predictor(predictor_loc)

    size = 128
    channels = 3
    feature_spec = read_tfrecord.matrix_label_feature_spec(size=size, channels=channels, float_label=True)

    # tfrecord_locs = ['/data/galaxy_zoo/decals/tfrecords/panoptes_featured_s128_lfloat_test.tfrecord']
    tfrecord_locs = ['/data/galaxy_zoo/decals/tfrecords/panoptes_featured_s128_lfloat_train.tfrecord']
    
    raw_subjects = read_tfrecord.load_examples_from_tfrecord(tfrecord_locs, feature_spec, n_examples=512)
    subjects = np.array([s['matrix'].reshape(size, size, channels) for s in raw_subjects])
    # plt.imshow(subjects[4].astype(int))
    # plt.savefig('temp_train.png')
    labels = np.array([s['label'] for s in raw_subjects])
    # print(labels[4])
    # exit(0)

    # fig, axes = read_tfrecord.show_examples(raw_subjects, size, channels)
    # fig.tight_layout()
    # fig.savefig('regression_train_examples.png')

    results_loc = 'results_train_d{}_local.txt'.format(dropout)

    # results = make_predictions.get_samples_of_subjects(model, subjects, n_samples=100)
    # np.savetxt(results_loc, results)

    results = np.loadtxt(results_loc)
    fig, axes = make_predictions.view_samples(results[:20], labels[:20])
    fig.tight_layout()
    fig.savefig('regression_train_d{}_local_inverse.png'.format(dropout))

    mse = metrics.mean_squared_error(labels, results.mean(axis=1))
    print('Mean mse: {}'.format(mse))

    baseline_mse = metrics.mean_squared_error(labels, np.ones_like(labels) * labels.mean())
    print('Baseline mse: {}'.format(baseline_mse))

    plt.figure()
    plt.hist(np.abs(results.mean(axis=1) - labels), label='Model', density=True, alpha=0.5)
    plt.hist(np.abs(labels.mean() - labels), label='Baseline', density=True, alpha=0.5)
    plt.legend()
    plt.xlabel('Mean Square Error')
    plt.savefig('mse_d{}_local_inverse.png'.format(dropout))

    plt.figure()
    save_loc = 'model_coverage_{}.png'.format(dropout)
    alpha_eval, coverage_at_alpha = dropout_calibration.check_coverage_fractions(results, labels)
    dropout_calibration.visualise_calibration(alpha_eval, coverage_at_alpha, save_loc)
