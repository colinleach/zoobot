
iteration_dir=/home/ubuntu/root/repos/zoobot/data/experiments/decals_smooth_may/iteration_1

estimator=1559567967

<!-- local_model="$iteration_dir/estimators/$estimator" -->
local_model="$iteration_dir/estimators" 

docker pull tensorflow/serving
docker run -p 8501:8501 --runtime=nvidia --mount type=bind,source=$local_model,target=/models/$estimator -e MODEL_NAME=$estimator -t tensorflow/serving