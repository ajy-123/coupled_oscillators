#!/bin/bash

#SBATCH --job-name=NVMakeDataset
#SBATCH --output=/home/ayeung_umass_edu/nv-nets/slurm/%j-%a.out
#SBATCH --error=/home/ayeung_umass_edu/nv-nets/slurm/%j-%a.err
#SBATCH -p cpu  # Partition
#SBATCH -c 10  # Number of Cores per Task
#SBATCH --mem=20192  # Requested Memory
#SBATCH --time=24:00:00
#SBATCH --array=1-10

module load miniconda/22.11.1-1
conda activate py310

# Run the script
cd /home/ayeung_umass_edu/nv-nets/experiments
python make_datasets.py \
--size 50 \