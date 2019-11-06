FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-0
# FROM gcr.io/deeplearning-platform-release/tf-gpu.1-14
# FROM tensorflow/tensorflow 
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

WORKDIR /home

RUN pip install --upgrade pip

# separately for speed, avoid re-installing with every new code version
ADD gz-panoptes-reduction/requirements.txt /home/gzreduction/requirements.txt
RUN pip install -r gzreduction/requirements.txt
ADD zoobot/requirements.txt /home/zoobot/requirements.txt
RUN pip install -r zoobot/requirements.txt

ADD shared-astro-utilities /home/shared-astro-utilities
RUN pip install -e shared-astro-utilities 

ADD gz-panoptes-reduction /home/gzreduction
RUN pip install -e gzreduction

ADD zoobot /home/zoobot
RUN pip install -e zoobot

WORKDIR /home/zoobot
