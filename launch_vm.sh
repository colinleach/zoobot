# Compute Engine Instance parameters
export IMAGE_FAMILY="common-container" 
export ZONE="us-east1-b"
export INSTANCE_NAME="zoobot_tf2"
export INSTANCE_TYPE="n1-standard-4"
export ACCELERATOR="type=nvidia-tesla-p4,count=1"

# see https://cloud.google.com/sdk/gcloud/reference/compute/instances/create
# https://cloud.google.com/compute/docs/instances/preemptible
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project="deeplearning-platform-release" \
        --maintenance-policy=TERMINATE \
        --accelerator=$ACCELERATOR \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=250GB \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --preemptible \
        --metadata="install-nvidia-driver=True,proxy-mode=project_editors,container=$IMAGE_URI"

# https://cloud.google.com/compute/docs/containers/deploying-containers