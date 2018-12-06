active-learning-regression works okay - current main branch for uncertainty feature
Latest definitely working commit (shown to yarin): a707679f84d749ffc5cf21a9f2e5760c81604351
(would tag this with regression-base, but now too late)
Loss immediately drops to 0.1 after 250 steps and then continues slowly dropping.

My own implementation of mean squared error was WRONG - the commit above only started working when I reverted to the built-in loss function.
The working commit with built-in loss function, at the start of al-binomial, is `binomial-base`.

Currently, am trying to work out how to get a binomial loss function working okay
(al-binomial).

The loss function now seems okay, but it quickly spikes and predictions go towards 1.
