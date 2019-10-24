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

docker rm $(docker ps -aq)

docker run -d \
    --name offline \
    -m 8GB \
    -v /Data/repos/zoobot/data:/home/zoobot/data \
    -v /Data/repos/zoobot/data/experiments/multilabel:/home/experiments/multilabel $IMAGE_URI  \
    python offline_training.py --train-dir /home/zoobot/data/decals/shards/multilabel_64/train --eval-dir /home/zoobot/data/decals/shards/multilabel_64/eval --experiment-dir /home/experiments/multilabel --epochs 10 --test

docker logs --follow offline


cd zoobot && git pull && cd ../ && cp zoobot/Dockerfile Dockerfile && docker build -f Dockerfile -t $IMAGE_URI ./ && docker rm $(docker ps -aq) && docker run -d \
    --name offline \
    -m 8GB \
    -v /Data/repos/zoobot/data:/home/zoobot/data \
    -v /Data/repos/zoobot/data/experiments/multilabel:/home/experiments/multilabel $IMAGE_URI  \
    python offline_training.py --train-dir /home/zoobot/data/decals/shards/multilabel_64/train --eval-dir /home/zoobot/data/decals/shards/multilabel_64/eval --experiment-dir /home/experiments/multilabel --shard-img-size 64 --epochs 10 --test

docker logs --follow offlinedocker run --name shards -v /home/data:/home/data gcr.io/zoobot-223419/zoobot python /home/zoobot/make_decals_tfrecords.py --labelled_catalog /home/data/bars_feat10_256.csv --eval-size 2500 --shard-dir /home/data/shards/multilabel_256 --img-size 256 docker run --name make_shards -v /home/data:/home/data gcr.io/zoobot-223419/zoobot python /home/zoobot/make_shards.py --labelled_catalog /home/data/bars_feat10_256.csv --eval-size 2500 --shard-dir /home/data/shards/multilabel_256 --img-size 256


docker push $IMAGE_URI

docker run -d \
    --name offline $IMAGE_URI  \
    python offline_training.py --train-dir /home/data/shards/multilabel_256/train --eval-dir /home/zoobot/data/decals/shards/multilabel_256/eval --experiment-dir /home/experiments/multilabel --shard-img-size 256 --epochs 10 --test

docker run --name shards -v /home/data:/home/data gcr.io/zoobot-223419/zoobot python /home/zoobot/make_decals_tfrecords.py --labelled_catalog /home/data/bars_feat10_256.csv --eval-size 2500 --shard-dir /home/data/shards/multilabel_256 --img-size 256 docker run --name make_shards -v /home/data:/home/data gcr.io/zoobot-223419/zoobot python /home/zoobot/make_shards.py --labelled_catalog /home/data/bars_feat10_256.csv --eval-size 2500 --shard-dir /home/data/shards/multilabel_256 --img-size 256