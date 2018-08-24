#/bin/sh!

# must already be logged in: docker login, username is mikewalmsley

# must be run from this directory
docker build -t=zoobot .
docker run -v ~/dockerlogs:/to_host zoobot /bin/sh -c "pytest zoobot;"​
# docker run -v ~/dockerlogs:/to_host zoobot /bin/sh -c "pytest zoobot/tests/read_tfrecord_test.py;"​
# docker run -v ~/dockerlogs:/to_host zoobot /bin/sh -c "pytest zoobot; cp -r * to_host"​
if [ $? -eq 0 ]; then 
    docker push mikewalmsley/zoobot:latest  # must be authenticated via `docker login`
fi  

echo 'Y' | docker system prune