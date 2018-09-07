# How to Run on EC2

## Setup

Make sure aws cli is installed:
`pip install aws`
`pip install awscli`
aws configure
(add ID, secret key, region=us-east-1, format=json)


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
- Re-run `aws configure` on local machine using an [active id](https://console.aws.amazon.com/iam/home?#/users/mikewalmsley?section=security_credentials)


## Zoobot Installation

From root...

Get the Zoobot directory from git
`git clone https://github.com/RustyPanda/zoobot.git && cd zoobot && git checkout bayesian-cnn`

Run the setup shell script. Downloads fits files and makes shards.
Downloading the native fits takes a few minutes (30mb/s, 6GB total for 7000 images) but is free.
`root=/home/ubuntu`
`source activate tensorflow_p36`
`pip install -r zoobot/requirements.txt`
`pip install -e $root/zoobot`
Variables are not preserved


Also log in to the S3 console:
`aws configure`


You can either make the shards directly, or download them from S3 (faster):

### Option A: Make Shards Directly 
`aws s3 cp s3://galaxy-zoo/decals/panoptes_predictions.csv $root/panoptes_predictions_original.csv`
`aws s3 sync s3://galaxy-zoo/decals/fits_native $root/fits_native`  # For now only the 7k we need. About 6GB.
`python $root/zoobot/zoobot/update_catalog_fits_loc.py`
`python $root/zoobot/zoobot/active_learning/make_shards.py $root $root/panoptes_predictions.csv`
Where the first arg is the directory into which to place the shard directory, and the second arg is the location of the catalog to use.
`shard_dir={path FROM ROOT to newly_created_shard_dir}`
Always re-upload to S3:
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

## Run Tensorboard to Monitor

On local machine, open an SSH tunnel to forward the ports using the `-L` flag:
`ssh -i $key -L 6006:127.0.0.1:6006 $user@$public_dns`

Then, via that SSH connection (or another), run
`tensorboard --logdir=.`
to run a Tensorboard server showing both baseline and real runs, if available


## Save results to S3

`aws s3 sync $root/$shard_dir s3://galaxy-zoo/active-learning/runs/ $run_dir`