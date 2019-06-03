
iteration_dir=/home/ubuntu/root/repos/zoobot/data/experiments/decals_smooth_may/iteration_1

estimator=1559567967

<!-- local_model="$iteration_dir/estimators/$estimator" -->
local_model="$iteration_dir/estimators" 

    docker pull tensorflow/serving:nightly-gpu

    docker run --runtime=nvidia  -p 8501:8501 --mount type=bind,source=$local_model,target=/models/$estimator -e MODEL_NAME=$estimator -t tensorflow/serving:nightly-gpu
    2019-06-03 14:26:11.679049: I tensorflow_serving/model_servers/server.cc:82] Building single TensorFlow model file config:  model_name: 1559567967 model_base_path: /models/1559567967