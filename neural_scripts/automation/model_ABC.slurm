#!/bin/bash
#
#SBATCH --job-name=training
#
#SBATCH --time=24:00:00
#SBATCH --gpus=1   
#SBATCH --partition=rotskoff,owners,gpu
#SBATCH -C GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --mem-per-gpu=32GB

#Print the node that is running this for later debugging purposes
echo "Current node: $SLURMD_NODENAME"
unset PMI_RANK

echo "Running polymer " $1
conda init bash
source /home/groups/rotskoff/emmit/anaconda3/bin/activate
conda activate polycomp_n

python -u FNO_model.py $1
