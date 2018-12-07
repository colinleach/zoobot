# cannot be run directly, copy instead!
# need to press 'yes' for now at clone, will pipe later
aws s3 cp s3://galaxy-zoo/github ~/.ssh/github  && \
aws s3 cp s3://galaxy-zoo/github.pub ~/.ssh/github.pub  && \
eval "$(ssh-agent -s)"  && \
chmod 400 ~/.ssh/github  && \
ssh-add ~/.ssh/github  && \
git clone git@github.com:mwalmsley/zoobot.git && cd zoobot && git checkout al-binomial && cd && \
pip install -r zoobot/requirements.txt && \
pip install -e zoobot  && \
git clone https://github.com/mwalmsley/shared-astro-utilities.git && \
pip install -e $root/shared-astro-utilities  && \
cd zoobot && dvc pull -r s3 && cd && \
echo "Environment ready, data fetched"
