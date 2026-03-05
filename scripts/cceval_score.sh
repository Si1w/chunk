#!/bin/bash
# Usage: sbatch scripts/cceval_score.sh [config]
#
# --- Job ---
#SBATCH -J cceval_score
#SBATCH -p cpu
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out

# --- Resources ---
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

CONFIG="${1:-configs/cceval.yaml}"

echo "=== Compute Scores ==="
echo "Config: ${CONFIG}"
echo "Start: $(date)"

uv run python -m eval.cceval.compute_score --config "${CONFIG}"

echo "=== Done: $(date) ==="
