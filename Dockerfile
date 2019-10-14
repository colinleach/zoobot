FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-0
# FROM tensorflow/tensorflow 

WORKDIR /home

ADD zoobot /home/zoobot
ADD shared-astro-utilities /home/shared-astro-utilities
ADD gz-panoptes-reduction /home/gzreduction

RUN pip install -r zoobot/requirements.txt
# will have tf2 from base image

RUN pip install -e zoobot
RUN pip install -e shared-astro-utilities 

WORKDIR /home/zoobot
