#!/bin/bash

# Set up fresh EC2 instance ready for shard creation
# Running on Deep Learning AMI or Miniconda with Python 3

# Run beforehand 'aws configure' and add secrets. Not in this script, obviously.

# not needed for DL AMI. Amazon Linux uses yum, Ubuntu uses apt-get
# sudo yum update
# sudo yum install git

# not needed for DL AMI. Pre-installed envs.
# conda update -n base conda
# conda create --name zoobot python=3.6
# source activate zoobot

# be in root
cd
# TODO get automatically
root=/home/ubuntu

source activate tensorflow_p36
pip install -r zoobot/requirements.txt  # needs C compiler for photutils, disabled for now

aws s3 cp s3://galaxy-zoo/decals/panoptes_predictions.csv $root/panoptes_predictions_original.csv
aws s3 sync s3://galaxy-zoo/decals/fits_native $root/fits_native  # gets everything, for now only the 7k we need. About 6GB.
