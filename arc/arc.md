vpn.ox.ac.uk
chri5177@ox.ac.uk
VPN password for remote access

ssh -X chri5177@oscgate.arc.ox.ac.uk
ssh -X chri5177@arcus-htc

sbatch script.sh
squeue -u chri5177

cd /data/phys-zooniverse/chri5177

Setup

ssh key is at /home/chri5177/.ssh/id_rsa
clone repo

module load python/anaconda3/2019.03

source activate /data/phys-zooniverse/chri5177/envs/zoobot

rsync -azv -e 'ssh -A -J chri5177@oscgate.arc.ox.ac.uk' /home/walml/repos/zoobot/data/latest_labelled_catalog.csv chri5177@arcus-htc:/data/phys-zooniverse/chri5177/repos/zoobot/data


sinfo -n arcus-htc-gpu023 --long

squeue -n arcus-htc-gpu023

GPU023 has 8 cpu and 2 P100s