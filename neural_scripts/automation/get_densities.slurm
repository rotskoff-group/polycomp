#!/bin/bash
#
#SBATCH --job-name=densities
#
#SBATCH --time=24:00:00
#SBATCH --gpus=1   
#SBATCH --partition=rotskoff,owners,gpu
#SBATCH -C GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --mem-per-gpu=32GB


#Print the node that is running this for later debugging purposes
echo "Current node: $SLURMD_NODENAME"
nvidia-smi
free -m
unset PMI_RANK

conda init bash
source /home/groups/rotskoff/emmit/anaconda3/bin/activate
conda activate polycomp_n


python -u ABC_densities.py ABC big_field.npy
python -u ABC_densities.py AB big_field.npy
python -u ABC_densities.py AC big_field.npy
python -u ABC_densities.py BC big_field.npy
python -u ABC_densities.py ABCCBA big_field.npy
python -u ABC_densities.py CAABBC big_field.npy

bash model_all.py
