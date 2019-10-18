# FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-0
FROM gcr.io/deeplearning-platform-release/tf-gpu.1-14
# FROM tensorflow/tensorflow 

WORKDIR /home

# separately for speed, avoid re-installing with every new code version
ADD gzreduction/requirements.txt /home/gzreduction/requirements.txt
RUN pip install -r zoobot/requirements.txt
ADD zoobot/requirements.txt /home/zoobot/requirements.txt
RUN pip install -r zoobot/requirements.txt
# will have tf from base image

ADD shared-astro-utilities /home/shared-astro-utilities
ADD gz-panoptes-reduction /home/gzreduction
ADD zoobot /home/zoobot

RUN pip install -e shared-astro-utilities 
RUN pip install -e gzreduction
RUN pip install -e zoobot

WORKDIR /home/zoobot
