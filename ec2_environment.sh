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

aws s3 cp s3://galaxy-zoo/decals/png_native.tar repos/zoobot/data/decals/png_native.tar


ZOOBOT_BRANCH=production_prototype
mkdir root && \
sudo mount /dev/xvdb root && \
cd root &&
eval "$(ssh-agent -s)"  && \
chmod 400 auth/github  && \
ssh-add auth/github  && \
cd repos/zoobot && git pull && git checkout $ZOOBOT_BRANCH && cd ../ && \
source activate tensorflow_p36 && \
pip install -r repos/zoobot/requirements.txt && \
pip install -e repos/zoobot  && \
pip install -r repos/shared-astro-utilities/requirements.txt && \
pip install -e repos/shared-astro-utilities  && \
pip install -r repos/gz-panoptes-reduction/requirements.txt && \
pip install -e repos/gz-panoptes-reduction  && \
source deactivate && \
screen -R run
