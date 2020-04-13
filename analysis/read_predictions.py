import ast

import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    with open('truth.txt', 'r') as f:
        truth = list(map(lambda x: float(x.strip()), f.readlines()))

    with open('predictions_no_pred_dropout.txt', 'r') as f:
        predictions = list(map(lambda x: ast.literal_eval(x.strip())['prediction'], f.readlines()))

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    ax1.hist(truth, density=True)
    ax1.set_xlabel('Truth')
    ax2.hist(predictions, density=True)
    ax2.set_ylabel('Predictions')
    fig.savefig('truth_vs_predictions_new_norm.png')
    
    fig, ax = plt.subplots()
    ax.scatter(truth, predictions[:len(truth)])
    ax.set_xlabel('truth')
    ax.set_ylabel('predictions')
    fig.savefig('truth_vs_predictions_new_norm_scatter.png')

