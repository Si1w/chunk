#!/bin/bash
# Fetch cceval dataset, clone repos, and build code windows.
# Usage: sbatch scripts/cceval_chunking.sh [config]
#
# --- Job ---
#SBATCH -J cceval_chunking
#SBATCH -p cpu
#SBATCH -t 12:00:00
#SBATCH -o %x_%j.out

# --- Resources ---
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

CONFIG="${1:-configs/cceval.yaml}"

echo "=== Fetch Dataset ==="
echo "Start: $(date)"

uv run python -m eval.cceval.fetch_dataset

echo "=== Chunking ==="
echo "Config: ${CONFIG}"

uv run python -m eval.cceval.make_window --config "${CONFIG}"

echo "=== Done: $(date) ==="
