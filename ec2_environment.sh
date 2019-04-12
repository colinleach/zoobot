# cannot be run directly, copy instead!
# need to press 'yes' for now at clone, will pipe later
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html
sudo mkfs -t ext4 /dev/xvdb && \
yes | git clone git@github.com:mwalmsley/zoobot.git && \
git clone https://github.com/mwalmsley/shared-astro-utilities.git && \
sudo chmod 777 root && cd root && \
aws s3 sync s3://galaxy-zoo/decals/fits_native data/fits_native && \
dvc pull -r s3 make_shards.dvc &&

aws s3 cp s3://galaxy-zoo/github auth/github  && \
aws s3 cp s3://galaxy-zoo/github.pub auth/github.pub  && \

dvc pull -r s3 repos/zoobot/data/gz2/gz2_classifications_and_subjects.csv
dvc pull -r s3 repos/zoobot/data/gz2/png.tar

dvc pull -r s3 data/decals/joint_catalog_selected_cols.csv.dvc
aws s3 cp s3://galaxy-zoo/decals/png_native.tar repos/zoobot/data/decals/png_native.tar

# BEFORE RUNNING THE BELOW COMMAND
# launch EC2 from sandbox AMI via Spot *in us-east-1b region*, m4.xlarge or similar
# set network group to open VPC (for now) to allow physics connections
# set IAM to Beanstalk (for s3 permissions)
# detach and delete 350gb volume (should remove this from the AMI)
# attach the volume working_volume to /dev/xvdb

ZOOBOT_BRANCH=production-prototype
sudo mount /dev/xvdb root && \
sudo chown -R ubuntu root
cd root &&
eval "$(ssh-agent -s)"  && \
chmod 400 auth/github  && \
ssh-add auth/github  && \
cd repos/zoobot && git fetch --all && git pull && git checkout $ZOOBOT_BRANCH && cd ../ && \
cd gz-panoptes-reduction && git pull && cd ../ && \
cd shared-astro-utilities && git pull && cd ../../ && \
source activate tensorflow_p36 && \
pip install -r repos/zoobot/requirements.txt && \
pip install -e repos/zoobot  && \
pip install -r repos/shared-astro-utilities/requirements.txt && \
pip install -e repos/shared-astro-utilities  && \
pip install -r repos/gz-panoptes-reduction/requirements.txt && \
pip install -e repos/gz-panoptes-reduction  && \
source deactivate && \
screen -R run

# if storage is a concern, can make a snapshot of and then delete working_volume
# $0.1/gb-month -> $35/month for the 350gb EBS volume, even when not attached! 
# <$17.5/month when stored as snapshot in s3, likely lower as not completely full