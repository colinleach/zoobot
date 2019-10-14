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

 gcloud compute instances create-with-container $INSTANCE_NAME \
     --container-image $IMAGE_URI \
     --zone=$ZONE \
     --maintenance-policy=TERMINATE \
     --accelerator=$ACCELERATOR \
     --machine-type=$INSTANCE_TYPE \
     --boot-disk-size=250GB \
     --preemptible

 gcloud compute instances create-with-container --container-image $IMAGE_URI $INSTANCE_NAME
     

gcloud beta compute --project=zoobot-223419 instances create-with-container instance-1 --zone=us-east1-b --machine-type=n1-standard-4 --subnet=default --network-tier=PREMIUM --metadata=google-logging-enabled=true --maintenance-policy=TERMINATE --service-account=6827612827-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --accelerator=type=nvidia-tesla-p100,count=1 --tags=http-server,https-server --image=cos-stable-77-12371-89-0 --image-project=cos-cloud --boot-disk-size=200GB --boot-disk-type=pd-standard --boot-disk-device-name=instance-1 --container-image=gcr.io/zoobot-223419/zoobot --container-restart-policy=always --labels=container-vm=cos-stable-77-12371-89-0 --reservation-affinity=any

gcloud compute --project=zoobot-223419 firewall-rules create default-allow-http --direction=INGRESS --priority=1000 --network=default --action=ALLOW --rules=tcp:80 --source-ranges=0.0.0.0/0 --target-tags=http-server

gcloud compute --project=zoobot-223419 firewall-rules create default-allow-https --direction=INGRESS --priority=1000 --network=default --action=ALLOW --rules=tcp:443 --source-ranges=0.0.0.0/0 --target-tags=https-server