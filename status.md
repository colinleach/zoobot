active-learning-regression works okay - current main branch for uncertainty feature
Latest definitely working commit (shown to yarin): a707679f84d749ffc5cf21a9f2e5760c81604351
(would tag this with regression-base, but now too late)
Loss immediately drops to 0.1 after 250 steps and then continues slowly dropping.

My own implementation of mean squared error was WRONG - the commit above only started working when I reverted to the built-in loss function.
The working commit with built-in loss function, at the start of al-binomial, is `binomial-base`.

Currently, am trying to work out how to get a binomial loss function working okay
(al-binomial).

186e6a8add26cd4915e047ba0bdbfa7570c848e5:
Using linear final layer, binomial loss + penalty term, trains for 4k to 0.23 RMSE while converging towards p0.5
Then goes a bit crazy (spike in loss), and then settles at p = 0.527
Repeat run (186e6a8add26cd4915e047ba0bdbfa7570c848e5_repeat) did not show the same convergence behaviour, but did show the same instability spikes.

7fa9629c8be18d6859485e19d9ae8c3f9acdb604:
Added L2 regularization term (to be consistent with the VI literature).
The weights were successfully reduced, but loss remained quite unstable. Mean RMSE 0.21.

0af785a10634bd26b25509b9933c3092ae2bd02c:
Use noisy labels: pick 0 or 1 in proportion with the vote fraction
Then use tf.nn.softmax_with_cross_entropy_v2 to squish and get a loss
(avoids worrying about binomial numerical instability problems)
Seems to be training okay so far, although prediction output on tensorboard is wrong (is actually first linear unit)


al-binomial-nonnoisy
Identical to al-bimomial commit above, except using deterministic labels
I think this reduces to the classification case that I've done before
Will be an interesting comparison in the nature of the predictions

al-binomial-4conv
Identical to al-binomial commit above, except with the third conv/pool layer duplicated (i.e. four conv layers).
Extra 2k epochs of training allowed, to allow for slower updates
Let's see what happens when we push deeper!
Don't get sucked into this - can always use NAS and then add dropout, if cross-entropy is enough
