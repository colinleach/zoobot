FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-0

# https://cloud.docker.com/u/mikewalmsley/repository/docker/mikewalmsley/zoobot
RUN gsutil cp s3://galaxy-zoo/credentials/github ~/.ssh/github
RUN gsutil cp s3://galaxy-zoo/credentials/github.pub ~/.ssh/github.pub
RUN chmod 400 ~/.ssh/github
RUN eval "$(ssh-agent -s)"  && ssh-add ~/.ssh/github

WORKDIR /home
RUN git clone git@github.com:mwalmsley/zoobot
RUN cd zoobot && git checkout al-iter-arms-smooth-full && cd ../
RUN git clone git@github.com:mwalmsley/gz-panoptes-reduction.git
RUN git clone git@github.com:mwalmsley/shared-astro-utilities.git

RUN pip install -r zoobot/requirements.txt
RUN pip install -e zoobot
RUN pip install -e shared-astro-utilities 

WORKDIR /home/zoobot
