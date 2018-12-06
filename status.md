active-learning-regression works okay - current main branch for uncertainty feature
Latest definitely working commit (shown to yarin): a707679f84d749ffc5cf21a9f2e5760c81604351
(would tag this with regression-base, but now too late)
Loss immediately drops to 0.1 after 250 steps and then continues slowly dropping.


mean squared error with l2 regularisation (mse_loss_with_l2 tag) seems to work well (although not complete)

Currently, am trying to work out how to get a binomial loss function working okay
(al-binomial).

The loss function now seems okay, but it quickly spikes and predictions go towards 1.
