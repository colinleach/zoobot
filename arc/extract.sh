#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --job-name=single_core
#SBATCH --ntasks-per-node=1
#SBATCH --partition=htc

module purge

# decals
# tar -xvzf /data/phys-zooniverse/chri5177/png_native.tar.gz -C /data/phys-zooniverse/chri5177

# gz2
tar -xvzf /data/phys-zooniverse/chri5177/gz2.tar -C /data/phys-zooniverse/chri5177
