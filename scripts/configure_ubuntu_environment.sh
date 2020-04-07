
# once only
# Create compute instances from Ubuntu 18.04 base image, or from Tensorflow GPU image
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

