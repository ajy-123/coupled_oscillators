#!/bin/bash

#SBATCH --job-name=MNIST_Classification_Subsampled
#SBATCH --output=/home/ayeung_umass_edu/nv-nets/slurm/%j-%a.out
#SBATCH --error=/home/ayeung_umass_edu/nv-nets/slurm/%j-%a.err
#SBATCH -p cpu-long  # Partition
#SBATCH -c 10  # Number of Cores per Task
#SBATCH --mem=80900  # Requested Memory
#SBATCH --time=4-12

module load miniconda/22.11.1-1
source /home/ayeung_umass_edu/.bashrc
conda activate py310

# Run the script
cd /home/ayeung_umass_edu/nv-nets/experiments
python mnist_classify_subsampled.py