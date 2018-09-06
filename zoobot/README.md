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
Add an EBS volume to allow for permanant storage. 
If using pre-existing shards, use that EBS snapshot.
Select existing security group `default` for home IP access

Save the .pem, if not already saved.

## Connect to instance

Copy public DNS (e.g. ec2-174-129-152-61.compute-1.amazonaws.com) from instance details
`ssh -i /path/my-key-pair.pem {ec2-user if plain linux, ubuntu if DL AMI}@{public DNS}`
The EC2 page has a handy button to create this command: press connect/standalone ssh client

If you get the error "Permission denied (publickey)"
- Check the username is `ubuntu`, not `ec2-user`
- Re-run `aws configure` on local machine using an [active id](https://console.aws.amazon.com/iam/home?#/users/mikewalmsley?section=security_credentials)


## Zoobot Installation

From root...

Get the Zoobot directory from git
`git clone https://github.com/RustyPanda/zoobot.git && cd zoobot && git checkout bayesian-cnn`

Run the setup shell script. Downloads fits files and makes shards.
Downloading the native fits takes a few minutes (30mb/s, 6GB total for 7000 images) but is free.
`zoobot/zoobot/active_learning/ec2_setup.sh`

## Make Shards

`python $root/zoobot/zoobot/update_catalog_fits_loc.py`
`python $root/zoobot/zoobot/active_learning/make_shards.py $root $root/panoptes_predictions.csv`
Where the first arg is the directory into which to place the shard directory, and the second arg is the location of the catalog to use.

## Run Active Learning

If shards are not already on the instance:
`shard_dir={desired_shard_dir}`, matching an S3 shard dir
`aws s3 sync s3://galaxy-zoo/active_learning/$shard_dir} $shard_dir`
Once shards are ready:
`python $root/zoobot/zoobot/active_learning/execute.py --shard_config=$shard_dir/shard_config.json --run_dir=$root/run`
where the first arg is the config object describing the shards, and the second is the directory to create run data (estimator, new tfrecords, etc).

## Run Tensorboard to Monitor

On local machine, open an SSH tunnel to forward the ports using the `-L` flag:
`ssh -i ~/mykeypair.pem -L 6006:127.0.0.1:6006 ubuntu@ec2-###-##-##-###.compute-1.amazonaws.com`

Then, via that SSH connection (or another), run
`tensorboard --logdir=.`
to run a Tensorboard server showing both baseline and real runs, if available


