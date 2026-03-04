#!/bin/bash
# One-time setup: pull the NVIDIA PyTorch container image.
#
# Usage: bash scripts/setup_container.sh
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
