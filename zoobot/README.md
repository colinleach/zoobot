# How to Run on EC2

## Setup

<!-- Make sure aws cli is installed: -->
<!-- `pip install aws` -->
<!-- `pip install awscli` -->
<!-- `aws configure` (add ID, secret key, region=us-east-1, format=json) -->

Make sure the instance has an IAM role with S3 read/write permissions attached.
This can be done post-launch.

## Launch an instance. 

Use AMI Miniconda/Py3 or Deep Learning AMI (inc. Tensorflow).
Cheapest supported instance is t2.small, or t3.small. Be sure to disable unlimited bursting!
Add an EBS volume to allow for permanant storage, if desired. S3 may be cheaper though. 
If using pre-existing shards, use that EBS snapshot.
Select existing security group `default` for home IP access

Save the .pem, if not already saved.

## Connect to instance

**Set up convience shell variables**

Copy public DNS from instance details

`public_dns={e.g. ec2-174-129-152-61.compute-1.amazonaws.com}`  

`key=/path/my-key-pair.pem`

`user=ubuntu` (or ec2-user if plain linux rather than DL AMI)


**Connect**

`ssh -i $key $user@$public_dns`

If you get the error "Permission denied (publickey)"
- Check the username is `ubuntu`, not `ec2-user`, or vica versa
<!-- - Re-run `aws configure` on local machine using an [active id](https://console.aws.amazon.com/iam/home?#/users/mikewalmsley?section=security_credentials). Choose us-east-1 as region and json as output format. -->


## Zoobot Installation

`source activate tensorflow_p36`

From root...

Get the Zoobot directory from git

Ensure key is in s3 (locally)
aws s3 cp ~/.ssh/github s3://galaxy-zoo/github
aws s3 cp ~/.ssh/github.pub s3://galaxy-zoo/github.pub

aws s3 cp s3://galaxy-zoo/github ~/.ssh/github
aws s3 cp s3://galaxy-zoo/github.pub ~/.ssh/github.pub
eval "$(ssh-agent -s)"
chmod 400 ~/.ssh/github
ssh-add ~/.ssh/github


`eval "$(ssh-agent -s)"`
<!-- `git clone https://github.com/mwalmsley/zoobot.git && cd zoobot && git checkout bayesian-cnn` -->
`git clone git@github.com:mwalmsley/zoobot.git && cd zoobot && git checkout al-binomial && cd`

Run the setup shell script. Downloads fits files and makes shards.
Downloading the native fits takes a few minutes (30mb/s, 6GB total for 7000 images) but is free.
`root=/home/ubuntu`
`pip install -r zoobot/requirements.txt`
`pip install -e zoobot`
<!-- extra requirement not on pypi -->
`git clone https://github.com/mwalmsley/shared-astro-utilities.git`
`pip install -e shared-astro-utilities`


# Basic Split 

Download the tfrecords:

<!-- `aws s3 sync s3://galaxy-zoo/basic-split/float data` -->
`cd zoobot && dvc pull && cd`

Run the regressor:

`python zoobot/run_zoobot_on_panoptes.py --ec2=True`

You can either make the shards directly, or download them from S3 (faster):


# Active Learning

## Get Fits (for new shards)
TODO: Dynamically, for when a minority become labelled, and then delete. Free!
`aws s3 sync s3://galaxy-zoo/decals/fits_native $root/fits_native`  For now only the 7k we need. About 6GB.


## Get Shards

### Option A: Make Shards Directly 

Get catalog with latest labels

`aws s3 cp s3://galaxy-zoo/decals/panoptes_predictions.csv $root/panoptes_predictions_original.csv`

Update catalog to point fits_loc at local fits_native folder

`python $root/zoobot/zoobot/update_catalog_fits_loc.py`

Create shards using make_shards.py ShardConfig defaults and current catalog

`python $root/zoobot/zoobot/active_learning/make_shards.py --base_dir=$root --catalog=$root/panoptes_predictions.csv`

Where the first arg is the directory into which to place the shard directory, and the second arg is the location of the catalog to use.

Update `shard_dir` shell variable

`shard_dir={path FROM ROOT to newly_created_shard_dir, automatically named based on ShardConfig}`

Re-upload to S3

`aws s3 sync $root/$shard_dir s3://galaxy-zoo/active-learning/$shard_dir`

### Option B: Download from S3
`shard_dir={desired_shard_dir}`, matching an S3 shard dir e.g. `shards_si64_sf28_l0.4`
`aws s3 sync s3://galaxy-zoo/active-learning/$shard_dir $root/$shard_dir`


## Run Active Learning

Once shards are ready:

`run_dir = {run dir relative to root}`

`python $root/zoobot/zoobot/active_learning/execute.py --shard_config=$root/$shard_dir/shard_config.json --run_dir=$root/$run_dir`

shard_config is the config object describing the shards. run_dir is the directory to create run data (estimator, new tfrecords, etc).
Optionally, add --baseline=True to select samples for labelling randomly.
**Check that logs are being recorded**. They should be in the directory the script was run from (i.e. root).

## Optional: Run Tensorboard to Monitor

On local machine, open an SSH tunnel to forward the ports using the `-L` flag:
`ssh -i $key -L 6006:127.0.0.1:6006 $user@$public_dns`

Then, via that SSH connection (or another), run
`source activate tensorflow_p36`
`tensorboard --logdir=.`
to run a Tensorboard server showing both baseline and real runs, if available


## Optional: Save results to S3
`aws s3 sync $root/$shard_dir s3://galaxy-zoo/active-learning/runs/$run_dir`