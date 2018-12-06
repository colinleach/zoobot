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
Then goes a bit crazy, and then settles at p = 0.527

