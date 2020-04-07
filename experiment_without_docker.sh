


# GCP:
# Generic useful commands
gcloud compute instances list
gcloud compute instances describe zoobot-p100-cli
gcloud compute ssh zoobot-p100-cli -- -L 8080:127.0.0.1:8080 -L 6006:127.0.0.1:6006
gcloud compute instances start zoobot-p100-cli
gcloud compute instances stop zoobot-p100-cli
gcloud compute instances delete zoobot-p100-cli

# once only
# START FROM tensorflow-gpu base! Also no CUDA worries :) Make new boot disk
export PYTHON=python3
gsutil cp -r gs://galaxy-zoo/credentials/* ~/.ssh
sudo chmod 600 ~/.ssh/github && eval "$(ssh-agent -s)" && ssh-add ~/.ssh/github
git clone git@github.com:mwalmsley/zoobot
export BRANCH=tf2
cd zoobot && git pull && git checkout $BRANCH && cd ../
rm -r zoobot/data  # to mount later
git clone git@github.com:mwalmsley/gz-panoptes-reduction
git clone git@github.com:mwalmsley/shared-astro-utilities

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda_install.sh
chmod u+x miniconda_install.sh
./miniconda_install.sh
# and follow prompts (including close and re-open)
conda create --name zoobot
source activate zoobot
conda install pip
pip install --upgrade pip
pip install --upgrade setuptools
pip install -r gz-panoptes-reduction/requirements.txt
pip install -r zoobot/requirements.txt
pip install -r shared-astro-utilities/requirements.txt
pip install -e shared-astro-utilities 
pip install -e gz-panoptes-reduction
pip install -e zoobot

# follow CUDA instructions for Ubuntu 18.04 https://www.tensorflow.org/install/gpu#install_cuda_with_apt

pip install tensorflow-gpu
pip install tensorflow-addons
# $PYTHON -m pip install -r gz-panoptes-reduction/requirements.txt
# $PYTHON -m pip install -r zoobot/requirements.txt
# $PYTHON -m pip install -r shared-astro-utilities/requirements.txt
# $PYTHON -m pip install -e shared-astro-utilities 
# $PYTHON -m pip install -e gz-panoptes-reduction
# $PYTHON -m pip install -e zoobot

export DATA_DIR=~/external # relative to home directory
DEVICE_LOC=/dev/sdb
sudo mkdir -p $DATA_DIR
sudo mount -o discard,defaults $DEVICE_LOC $DATA_DIR
sudo chmod a+w $DATA_DIR


echo {"username": "mikewalmsley", "password": "SET ME" > ~/gz-panoptes-reduction/gzreduction/panoptes/api/zooniverse_login.json


export PNG_PREFIX=$DATA_DIR
export SHARD_IMG_SIZE=256
export SHARD_MAX=''
export SHARD_EVAL='2500'

# local:
export PYTHON=python
export DATA_DIR=/home/repos/zoobot/data
export PNG_PREFIX=/Volumes/alpha
export SHARD_IMG_SIZE=64
export SHARD_MAX='--max=1000'
export SHARD_EVAL='500'

# shared
export SHARD_DIR=$DATA_DIR/decals/shards/multilabel_feat10_$SHARD_IMG_SIZE
export LABELLED_CATALOG=$DATA_DIR/prepared_catalogs/mac_catalog_feat10_correct_labels_full_256.csv
export EXPERIMENT_DIR=$DATA_DIR/experiments/multiquestion_smooth_spiral_feat10_tf2_$SHARD_IMG_SIZE
export EPOCHS=2

export BRANCH=tf2
cd zoobot && git pull && git checkout $BRANCH && cd ../

# make test shards (if needed)
python zoobot/make_decals_tfrecords.py --labelled-catalog $DATA_DIR/decals/prepared_catalogs/decals_smooth_may/labelled_catalog.csv  --shard-dir=$SHARD_DIR --img-size=$SHARD_IMG_SIZE --eval-size=$SHARD_EVAL $SHARD_MAX --png-prefix=$PNG_PREFIX

# run training
python zoobot/offline_training.py --experiment-dir $EXPERIMENT_DIR --train-dir $SHARD_DIR/train --eval-dir $SHARD_DIR/eval --shard-img-size=$SHARD_IMG_SIZE --epochs $EPOCHS