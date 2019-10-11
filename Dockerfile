FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-0
# FROM tensorflow/tensorflow 

# https://cloud.docker.com/u/mikewalmsley/repository/docker/mikewalmsley/zoobot
# RUN gsutil cp s3://galaxy-zoo/credentials/github ~/.ssh/github
# RUN gsutil cp s3://galaxy-zoo/credentials/github.pub ~/.ssh/github.pub
# RUN ls credentials
# RUN ls ~/.ssh
# COPY credentials/github ~/.ssh/github
# COPY credentials/github.pub ~/.ssh/github.pub
# RUN pwd
# RUN ls /home
# RUN chmod 400 ~/.ssh/github
# RUN eval "$(ssh-agent -s)"  && ssh-add ~/.ssh/github

# env variable but only during build
ARG GIT_TOKEN

WORKDIR /home
# ADD credentials  /home/credentials

# RUN echo “[url \”git@github.com:\”]\n\tinsteadOf = https://github.com/" >> /root/.gitconfig

RUN mkdir /root/.ssh
RUN git config --global url."https://$GIT_TOKEN:@github.com/".insteadOf "https://github.com/"

# Skip Host verification for git
RUN echo "StrictHostKeyChecking no " > /root/.ssh/config
# RUN eval "$(ssh-agent -s)"  && ssh-add /home/credentials/github

RUN git clone git@github.com:mwalmsley/zoobot
RUN cd zoobot && git checkout al-iter-arms-smooth-full && cd ../
RUN git clone git@github.com:mwalmsley/gz-panoptes-reduction.git
RUN git clone git@github.com:mwalmsley/shared-astro-utilities.git

RUN pip install -r zoobot/requirements.txt
# # will have tf2 from base image
# # RUN pip install tensorflow
RUN pip install -e zoobot
RUN pip install -e shared-astro-utilities 

WORKDIR /home/zoobot
