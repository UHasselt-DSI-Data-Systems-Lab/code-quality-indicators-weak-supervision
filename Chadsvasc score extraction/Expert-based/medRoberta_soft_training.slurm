#!/bin/bash
#SBATCH --job-name=train_medroberta_soft
#SBATCH --account=add_account_name_here
#SBATCH --output=slurm_files/train_medroberta_maj_prior_%j.out
#SBATCH --error=slurm_files/train_medroberta_maj_prior_%j.err
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# -----------------------------
# Activate conda environment
# -----------------------------
ENV_PATH="$VSC_DATA/envs/laat"
eval "$($VSC_DATA/miniconda3/bin/conda shell.bash hook)"
conda activate $ENV_PATH

# -----------------------------
# Move to repo folder
# -----------------------------
cd "../code/ChadsVasc score extraction"
export PYTHONPATH=$PWD:$PYTHONPATH

# -----------------------------
# Run training
# -----------------------------
python medRoberta_soft_training_sliding_window.py


conda deactivate
