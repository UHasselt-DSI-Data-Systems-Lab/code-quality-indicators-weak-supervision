#!/bin/bash
#SBATCH --job-name=rag_approach
#SBATCH --account=add your account
#SBATCH --output=slurm_files/geitje_%j.out
#SBATCH --error=slurm_files/geitje_%j.err
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

module purge
module load cluster/wice/gpu
module load GCC/12.3.0
module load CUDA/12.6.0

# -----------------------------
# Activate Conda
# -----------------------------
ENV_PATH="$VSC_DATA/envs/RAG"

eval "$($VSC_DATA/miniconda3/bin/conda shell.bash hook)"
conda activate "$ENV_PATH"

# Forceer een herinstallatie van de nieuwste versie met CUDA support
# CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
# python -c "import llama_cpp; print(llama_cpp.__version__)"
# -----------------------------
# Ensure correct llama-cpp build for H100 (Hopper = compute 90)
# Do this once — not every job if possible
# -----------------------------
# Uncomment if you need reinstall inside job

# pip uninstall -y llama-cpp-python

# CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=90-real" \
# FORCE_CMAKE=1 \
# pip install llama-cpp-python --no-cache-dir --force-reinstall

# -----------------------------
# Move to project folder
# -----------------------------
cd "/code/ChadsVasc score extraction/Contextual LLM"

export PYTHONPATH="$PWD:$PYTHONPATH"

# -----------------------------
# Diagnostics
# -----------------------------
# echo "===== GPU INFO ====="
# nvidia-smi

# echo "===== Torch GPU Capability ====="
# python - << EOF
# import torch
# print("Torch GPU capability:", torch.cuda.get_device_capability())
# print("Torch device:", torch.cuda.get_device_name(0))
# EOF

# echo "===== LlamaCPP Version ====="
# python - << EOF
# import llama_cpp
# print(llama_cpp.__version__)
# EOF

# -----------------------------
# Run pipeline
# -----------------------------
python configuration_geitje.py

conda deactivate
