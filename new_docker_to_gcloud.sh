#!/usr/bin/env bash
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=zoobot
export IMAGE_TAG=latest
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
# gcr.io/zoobot-223419/zoobot:latest

cd zoobot && git pull && cd ../ && cp zoobot/Dockerfile Dockerfile && docker build -f Dockerfile -t $IMAGE_URI ./

export SHARD_IMG_SIZE=64


docker rm $(docker ps -aq)

# Build shards locally (needed rarely)

docker run  \
    --rm \
    --name shards \
    -v /Volumes/alpha/decals:/home/data/decals \
    -v /Data/repos/zoobot/data:/home/zoobot/data \
    -v /Data/repos/zoobot/data/experiments/multilabel_$SHARD_IMG_SIZE:/home/experiments/multilabel_$SHARD_IMG_SIZE $IMAGE_URI  \
    python make_decals_tfrecords.py --labelled-catalog /home/zoobot/data/decals/prepared_catalogs/decals_smooth_may/labelled_catalog.csv --eval-size=500 --shard-dir=/home/zoobot/data/decals/shards/multilabel_$SHARD_IMG_SIZE --img-size=$SHARD_IMG_SIZE --max=1000 --png-prefix=/home/data

# run locally
docker run \
    --rm \
    --name offline \
    -m 8GB \
    -v /Data/repos/zoobot/data:/home/zoobot/data \
    -v /Data/repos/zoobot/data/experiments/multilabel_$SHARD_IMG_SIZE:/home/experiments/multilabel_$SHARD_IMG_SIZE $IMAGE_URI  \
    python offline_training.py --train-dir /home/zoobot/data/decals/shards/multilabel_$SHARD_IMG_SIZE/train --eval-dir /home/zoobot/data/decals/shards/multilabel_$SHARD_IMG_SIZE/eval --experiment-dir /home/experiments/multilabel_$SHARD_IMG_SIZE --shard-img-size $SHARD_IMG_SIZE --epochs 10 --test


# pr pull, build, and run locally in one command
cd zoobot && git pull && cd ../ && cp zoobot/Dockerfile Dockerfile && docker build -f Dockerfile -t $IMAGE_URI ./ && docker run \
    --rm \
    --name offline \
    -m 8GB \
    -v /Data/repos/zoobot/data:/home/zoobot/data \
    -v /Data/repos/zoobot/data/experiments/multilabel_$SHARD_IMG_SIZE:/home/experiments/multilabel_$SHARD_IMG_SIZE $IMAGE_URI  \
    python offline_training.py --train-dir /home/zoobot/data/decals/shards/multilabel_$SHARD_IMG_SIZE/train --eval-dir /home/zoobot/data/decals/shards/multilabel_$SHARD_IMG_SIZE/eval --experiment-dir /home/experiments/multilabel_$SHARD_IMG_SIZE --shard-img-size $SHARD_IMG_SIZE --epochs 10 --test

# if running in detached [-d] mode
docker logs --follow offline

# construct shards with full catalog (locally)
# requires catalog input files
docker run --name shards -v /home/data:/home/data gcr.io/zoobot-223419/zoobot python /home/zoobot/make_decals_tfrecords.py --labelled_catalog /home/data/bars_feat10_$SHARD_IMG_SIZE.csv --eval-size 2500 --shard-dir /home/data/shards/multilabel_$SHARD_IMG_SIZE --img-size $SHARD_IMG_SIZE docker run --name make_shards -v /home/data:/home/data gcr.io/zoobot-223419/zoobot python /home/zoobot/make_shards.py --labelled_catalog /home/data/bars_feat10_$SHARD_IMG_SIZE.csv --eval-size 2500 --shard-dir /home/data/shards/multilabel_$SHARD_IMG_SIZE --img-size $SHARD_IMG_SIZE


docker push $IMAGE_URI

docker run -d \
    --name offline $IMAGE_URI  \
    python offline_training.py --train-dir /home/data/shards/multilabel_$SHARD_IMG_SIZE/train --eval-dir /home/zoobot/data/decals/shards/multilabel_$SHARD_IMG_SIZE/eval --experiment-dir /home/experiments/multilabel --shard-img-size $SHARD_IMG_SIZE --epochs 10 --test

docker run --name shards -v /home/data:/home/data gcr.io/zoobot-223419/zoobot python /home/zoobot/make_decals_tfrecords.py --labelled_catalog /home/data/bars_feat10_$SHARD_IMG_SIZE.csv --eval-size 2500 --shard-dir /home/data/shards/multilabel_$SHARD_IMG_SIZE --img-size $SHARD_IMG_SIZE docker run --name make_shards -v /home/data:/home/data gcr.io/zoobot-223419/zoobot python /home/zoobot/make_shards.py --labelled_catalog /home/data/bars_feat10_$SHARD_IMG_SIZE.csv --eval-size 2500 --shard-dir /home/data/shards/multilabel_$SHARD_IMG_SIZE --img-size $SHARD_IMG_SIZE

!python /home/zoobot/make_decals_tfrecords.py --labelled_catalog /home/data/bars_feat10_{shard_img_size}.csv --eval-size 2500 --shard-dir /home/data/shards/multilabel_{shard_img_size} --img-size {shard_img_size} docker run --name make_shards -v /home/data:/home/data gcr.io/zoobot-223419/zoobot python /home/zoobot/make_shards.py --labelled_catalog /home/data/bars_feat10_{shard_img_size}.csv --eval-size 2500 --shard-dir /home/data/shards/multilabel_{shard_img_size} --img-size {shard_img_size}

# could change back to container-optimised OS
# Compute Engine Instance parameters
export ZONE="us-east1-c"
export INSTANCE_NAME="zoobot-p100-cli"
export INSTANCE_TYPE="n1-standard-4"
export ACCELERATOR="type=nvidia-tesla-p100,count=1"

export IMAGE_PROJECT="ubuntu-os-cloud"
export IMAGE_FAMILY="ubuntu-1804-lts"
# OR
# export IMAGE_PROJECT="cos-cloud"
# export IMAGE_FAMILY="cos-stable"
# https://cloud.google.com/compute/docs/images#unshielded-images
# image-family will use the latest non-deprecated image of that family
# image-project is needed to avoid looking for the image under 'zoobot'
export METADATA_FROM_FILE="cos-gpu-installer-env=scripts/gpu-installer-env,user-data=install-test-gpu.cfg,run-installer-script=scripts/run_installer.sh,run-cuda-test-script=scripts/run_cuda_test.sh"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --maintenance-policy=TERMINATE \
        --accelerator=$ACCELERATOR \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --scopes=https://www.googleapis.com/auth/cloud-platform
        # --metadata='install-nvidia-driver=True'
        #         --metadata-from-file=$METADATA_FROM_FILE \
        # --scopes=https://www.googleapis.com/auth/cloud-platform
        # --scopes='storage-full'
        # not necessary to change from default service account because I gave it storage admin rights

        # --metadata="install-nvidia-driver=True" doesn't seem to do anything, perhaps only has effect on some instances?



                # --image-family=common-cu101 \
        # --image-project=deeplearning-platform-release \

# Generic useful commands
gcloud compute instances list
gcloud compute instances describe zoobot-p100-cli
gcloud compute ssh zoobot-p100-cli -- -L 8080:127.0.0.1:8080
gcloud compute instances start zoobot-p100-cli
gcloud compute instances stop zoobot-p100-cli
gcloud compute instances delete zoobot-p100-cli

# Setting up base image to run CUDA-enabled Docker images

# Install docker
# https://docs.docker.com/install/linux/docker-ce/ubuntu/
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common &&
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - &&
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable" &&
sudo apt-get update &&
sudo apt-get install docker-ce docker-ce-cli containerd.io

# Authenticate GCR
# https://cloud.google.com/container-registry/docs/advanced-authentication
gcloud auth configure-docker
# may need to re-install gcloud via 
# https://cloud.google.com/sdk/docs/quickstart-linux
docker pull gcr.io/zoobot-223419/zoobot:latest

# Install CUDA
# https://devblogs.nvidia.com/gpu-containers-runtime/
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda

# Install nvidia-docker
# https://nvidia.github.io/nvidia-docker/
sudo apt-get remove --purge nvidia-cuda*
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
sudo nvidia-container-cli --load-kmods info

# Check it works
sudo docker run --rm --runtime=nvidia -ti nvidia/cuda

# TODO attach disk with CLI

# Mount disk
# https://cloud.google.com/compute/docs/disks/add-persistent-disk
sudo lsblk (should be at sdb)  # optional
DEVICE_LOC=/dev/sdb
MNT_DIR=/mnt/disks/data
sudo mkdir -p $MNT_DIR
sudo mount -o discard,defaults $DEVICE_LOC $MNT_DIR
sudo chmod a+w $MNT_DIR

# Run default notebook/tensorboard
export SHARD_IMG_SIZE=256
export MNT_DIR=/mnt/disks/data
docker run -d --runtime=nvidia -v $MNT_DIR:/home/data -p 8080:8080 gcr.io/zoobot-223419/zoobot:latest 

# all galaxies
export SHARD_DIR=/home/data/decals/shards/multilabel_all_$SHARD_IMG_SIZE
export LABELLED_CATALOG=/home/data/prepared_catalogs/decals_smooth_may/labelled_catalog.csv
# OR
# feat10 only
export SHARD_DIR=/home/data/decals/shards/multilabel_feat10_$SHARD_IMG_SIZE
export LABELLED_CATALOG=/home/data/prepared_catalogs/mac_catalog_feat10_correct_labels_full_256.csv

# make shards
docker run -d --runtime=nvidia -v $MNT_DIR:/home/data gcr.io/zoobot-223419/zoobot:latest python make_decals_tfrecords.py --labelled-catalog $LABELLED_CATALOG --eval-size=2500 --shard-dir=$SHARD_DIR --img-size=$SHARD_IMG_SIZE --png-prefix=/home/data

# export EXPERIMENT_DIR=/home/data/experiments/multilabel_feat10_$SHARD_IMG_SIZE
export EXPERIMENT_DIR=/home/data/experiments/multiquestion_smooth_spiral_feat10_$SHARD_IMG_SIZE
export EPOCHS=100
docker run -d --runtime=nvidia -v $MNT_DIR:/home/data gcr.io/zoobot-223419/zoobot:latest python offline_training.py --experiment-dir $EXPERIMENT_DIR --train-dir $SHARD_DIR/train --eval-dir $SHARD_DIR/eval --shard-img-size=$SHARD_IMG_SIZE --epochs $EPOCHS
