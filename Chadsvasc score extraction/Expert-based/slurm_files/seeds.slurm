#!/bin/bash
#SBATCH --job-name=seeds_cardio
#SBATCH --account=add_account_name_here
#SBATCH --output=slurm_files/seeds_cardioberta_%j.out
#SBATCH --error=slurm_files/seeds_cardioberta_%j.err
#SBATCH --clusters=wice
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

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
python soft_training_seeds.py --model_name "UMCU/CardioBERTa.nl_clinical"


conda deactivate
