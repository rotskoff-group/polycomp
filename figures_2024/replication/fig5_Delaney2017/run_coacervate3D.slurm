#!/bin/bash
#
#SBATCH --job-name=CART_SCAN
#
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --partition=gpu,hns,owners,rotskoff
#SBATCH --array=0-24

#Print the node that is running this for later debugging purposes
echo "Current node: $SLURMD_NODENAME"
unset PMI_RANK


conda init bash
source /home/groups/rotskoff/emmit/anaconda3/bin/activate
conda activate gpu_gibbs

python -u run_coacervate3D_log.py $SLURM_ARRAY_TASK_MAX $SLURM_ARRAY_TASK_ID 
