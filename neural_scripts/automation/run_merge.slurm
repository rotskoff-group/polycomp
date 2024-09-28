#!/bin/bash
#
#SBATCH --job-name=merge
#
#SBATCH --time=0:20:00
#SBATCH --gpus=1   
#SBATCH --partition=rotskoff,owners,gpu
#SBATCH -C GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --mem-per-gpu=32GB

#Print the node that is running this for later debugging purposes
echo "Current node: $SLURMD_NODENAME"
unset PMI_RANK

conda init bash
source /home/groups/rotskoff/emmit/anaconda3/bin/activate
conda activate polycomp_n

python -u merge_shuffle.py

sbatch get_densities.slurm
