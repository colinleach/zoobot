#!/usr/bin/env bash
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=zoobot
export IMAGE_TAG=latest
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f Dockerfile -t $IMAGE_URI ./

# first time only
# gcloud auth configure-docker  

# docker push $IMAGE_URI

# https://cloud.google.com/blog/products/ai-machine-learning/introducing-deep-learning-containers-consistent-and-portable-environments
# https://cloud.docker.com/u/mikewalmsley/repository/docker/mikewalmsley/zoobot

cd zoobot && git pull && cd ../ && cp zoobot/Dockerfile Dockerfile && docker build -f Dockerfile -t $IMAGE_URI ./

export SHARD_IMG_SIZE=64

docker rm $(docker ps -aq)

# run locally
docker run -d \
    --name offline \
    -m 8GB \
    -v /Data/repos/zoobot/data:/home/zoobot/data \
    -v /Data/repos/zoobot/data/experiments/multilabel_$SHARD_IMG_SIZE:/home/experiments/multilabel_$SHARD_IMG_SIZE $IMAGE_URI  \
    python offline_training.py --train-dir /home/zoobot/data/decals/shards/multilabel_$SHARD_IMG_SIZE/train --eval-dir /home/zoobot/data/decals/shards/multilabel_$SHARD_IMG_SIZE/eval --experiment-dir /home/experiments/multilabel_$SHARD_IMG_SIZE --epochs 10 --test


# pr pull, build, and run locally in one command
cd zoobot && git pull && cd ../ && cp zoobot/Dockerfile Dockerfile && docker build -f Dockerfile -t $IMAGE_URI ./ && docker rm $(docker ps -aq) ; docker run -d \
    --name offline \
    -m 8GB \
    -v /Data/repos/zoobot/data:/home/zoobot/data \
    -v /Data/repos/zoobot/data/experiments/multilabel_$SHARD_IMG_SIZE:/home/experiments/multilabel_$SHARD_IMG_SIZE $IMAGE_URI  \
    python offline_training.py --train-dir /home/zoobot/data/decals/shards/multilabel_$SHARD_IMG_SIZE/train --eval-dir /home/zoobot/data/decals/shards/multilabel_$SHARD_IMG_SIZE/eval --experiment-dir /home/experiments/multilabel_$SHARD_IMG_SIZE --shard-img-size $SHARD_IMG_SIZE --epochs 10 --test


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