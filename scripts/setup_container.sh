#!/bin/bash
# One-time setup: pull the NVIDIA PyTorch container image and sync dependencies.
#
# Usage: sbatch scripts/setup_container.sh
#
# --- SLURM directives ---
#SBATCH -J setup_container
#SBATCH -p cpu
#SBATCH -t 02:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G

set -euo pipefail

SIF_DIR="/scratch/users/${USER}/images"
SIF="${SIF_DIR}/pytorch_24.12.sif"

mkdir -p "${SIF_DIR}"

if [ -f "${SIF}" ]; then
    echo "Image already exists: ${SIF}"
else
    echo "Pulling NVIDIA PyTorch image (this takes a while)..."
    singularity pull "${SIF}" docker://nvcr.io/nvidia/pytorch:24.12-py3
fi

echo "Done: ${SIF}"

# Sync Python dependencies inside the container
# uv is on the host; bind it into the container so it can run
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UV_BIN="$(which uv)"

echo "Syncing dependencies..."
singularity exec \
    --bind "/scratch/users/${USER}:/scratch/users/${USER}" \
    --bind "${UV_BIN}:/usr/local/bin/uv:ro" \
    --pwd "${PROJECT_DIR}" \
    "${SIF}" uv sync --all-extras

echo "Done: uv sync"
