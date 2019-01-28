# cannot be run directly, copy instead!
# need to press 'yes' for now at clone, will pipe later
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html
sudo mkfs -t ext4 /dev/xvdb && \
yes | git clone git@github.com:mwalmsley/zoobot.git && \
git clone https://github.com/mwalmsley/shared-astro-utilities.git && \
sudo chmod 777 root && cd root && \
aws s3 sync s3://galaxy-zoo/decals/fits_native data/fits_native && \
dvc pull -r s3 make_shards.dvc &&

mkdir root && \
sudo mount /dev/xvdb root && \
aws s3 cp s3://galaxy-zoo/github ~/.ssh/github  && \
aws s3 cp s3://galaxy-zoo/github.pub ~/.ssh/github.pub  && \
eval "$(ssh-agent -s)"  && \
chmod 400 ~/.ssh/github  && \
ssh-add ~/.ssh/github  && \
cd root/zoobot && git pull && git checkout al-iter-shuffled && cd ../ && \
source activate tensorflow_p36 && \
pip install -r zoobot/requirements.txt && \
pip install -e zoobot  && \
pip install -e shared-astro-utilities  && \
cd zoobot && \
mkdir data/gz2_shards && \
sudo mount /dev/xvdc data/gz2_shards && \
source deactivate && \
screen -R run
