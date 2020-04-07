


# GCP:
# Generic useful commands
gcloud compute instances list
gcloud compute instances describe zoobot-p100-cli
gcloud compute ssh zoobot-p100-cli -- -L 8080:127.0.0.1:8080 -L 6006:127.0.0.1:6006
gcloud compute instances start zoobot-p100-cli
gcloud compute instances stop zoobot-p100-cli
gcloud compute instances delete zoobot-p100-cli
