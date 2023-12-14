#!/bin/bash
#
#SBATCH --job-name=CART_SCAN
#
#SBATCH --time=30:00:00
#SBATCH --gpus=1
#SBATCH --partition=rotskoff,owners,gpu,hns

#Print the node that is running this for later debugging purposes
echo "Current node: $SLURMD_NODENAME"
unset PMI_RANK


conda init bash
source /home/groups/rotskoff/emmit/anaconda3/bin/activate
conda activate gpu_gibbs

python -u coacervate_sherlock.py
