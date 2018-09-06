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
- Check the username is `ubun re-run aws configure with an active id.
https://console.aws.amazon.com/iam/home?#/users/mikewalmsley?section=security_credentials
