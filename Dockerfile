FROM python:3.6

WORKDIR /repos/zoobot

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD . .

RUN find . -name '*.pyc' -delete