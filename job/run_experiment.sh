#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --job-name=train
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --gpus=rtx5000:1
#SBATCH --mail-type=ALL

module load miniconda
conda activate cs7643
module load CUDA/10.2.89
python experiment/train.py
