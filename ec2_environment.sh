# cannot be run directly, copy instead!
# need to press 'yes' for now at clone, will pipe later
source activate tensorflow_p36 && \
aws s3 cp s3://galaxy-zoo/github ~/.ssh/github  && \
aws s3 cp s3://galaxy-zoo/github.pub ~/.ssh/github.pub  && \
eval "$(ssh-agent -s)"  && \
chmod 400 ~/.ssh/github  && \
ssh-add ~/.ssh/github  && \
git clone git@github.com:mwalmsley/zoobot.git && cd zoobot && git checkout al-binomial-4conv && cd && \
pip install -r zoobot/requirements.txt && \
pip install -e zoobot  && \
git clone https://github.com/mwalmsley/shared-astro-utilities.git && \
pip install -e shared-astro-utilities  && \
cd zoobot && dvc pull -r s3 && cd && \
source deactivate && \
screen -R run && \
source activate tensorflow_p36 && \
echo "Environment ready, data fetched"
