active-learning-regression works okay - current main branch for uncertainty feature
Latest definitely working commit (shown to yarin): a707679f84d749ffc5cf21a9f2e5760c81604351
(would tag this with regression-base, but now too late)
Loss immediately drops to 0.1 after 250 steps and then continues slowly dropping.



My own implementation of mean squared error was WRONG - the commit above only started working when I reverted to the built-in loss function.
The working commit with built-in loss function, at the start of al-binomial, is `binomial-base`.

Currently, am trying to work out how to get a binomial loss function working okay
(al-binomial).

2b61a9fdc1af01c10dfb1172b0bdf2e69fff10b2:
My poor intermediate attempts at getting the binomial loss to work
https://github.com/mwalmsley/zoobot/commit/2b61a9fdc1af01c10dfb1172b0bdf2e69fff10b2

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
Need to double-check if output predictions are correct
Reference datetime: 1544206935
https://s3.console.aws.amazon.com/s3/buckets/galaxy-zoo/basic-split/runs/0af785a10634bd26b25509b9933c3092ae2bd02c/1544206935/?region=us-east-1&tab=overview


c26de59ddf92389a3e692632a86fdbcf86035013:
As above (three layers), but with an additional L2 loss added. 
Results are missing??


## al-binomial-nonnoisy ##

Single commit: 923b347a982541f0962e69627968e47abe171500
Identical to al-bimomial commit above, except using deterministic labels
I think this reduces to the classification case that I've done before
Will be an interesting comparison in the nature of the predictions.
I'm happy with this baseline - no further work needed for now.
Ends up slightly overfitting
Reference datetime version: 1544202382
https://s3.console.aws.amazon.com/s3/buckets/galaxy-zoo/basic-split/runs/923b347a982541f0962e69627968e47abe171500/1544202382/?region=us-east-1&tab=overview


## al-binomial-4conv ##
Initially identical to al-binomial commit above, except with the third conv/pool layer duplicated (i.e. four conv layers).
Extra 2k epochs of training allowed, to allow for slower updates

f56cd3ae80a4a5d4b78bdf9a5615359ae98e3503:
Was supposed to be four layers, but I actually forgot to pull.
Is actually three layers (no L2) from 0af785a10634bd26b25509b9933c3092ae2bd02c, but with predictions fixed. Will rename.

16e16919db722588b2b66d48e014ff0ff6fd0cc5:
Third conv/pool layer duplicated (as above)
Results are similar to al-binomial.

b87b062a40f0ac01c5543cf0b1f1b32f44501b13:
Change filters from 128, 64, 64 64 to 32, 64, 128, 128 following traditional design of increasing filters with depth
Results are similar still, but speed is slightly improved

9eef344784b313cff4f848cf85dd65394ca24365:
Double up the first two conv layers (conv1, pool1, conv2, pool2 to conv1, conv1b, pool1, conv2, conv2b, pool2), following VGG16 pattern of several convs per pool
Excellent results - significant loss reduction!
Reference datetime: 1544381343
https://s3.console.aws.amazon.com/s3/buckets/galaxy-zoo/basic-split/runs/9eef344784b313cff4f848cf85dd65394ca24365/1544381343/?region=us-east-1&tab=overview

6a5f999ae18f8a1ef5f5099166fcd292112ba644:
Following success with doubling up the first two conv layers, also double up the third and fourth conv layers
Did not improve loss vs. 9eef34

65898b9: 
Revert the extra third and fourth doubled layers
Add dropout between every conv layer, to be consistent with literature (although outside Bayesian realm, this is unusual)
(final consistency step will be to add L2 loss back in)
Important: change dropout_on switch to also put dropout on at predict time. Amazingly, this was not already set. Could have major implications.
Training ground to a halt at dropout = 0.5.

c2548d0:
Dropout on early layers prevented, or massively slowed, learning. Possibly the flag change may have been involved.
I've reduced the dropout rate on early layers to the dense layer / 10. Here, rate = 0.05.
Running now.



