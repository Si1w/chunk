#!/bin/bash
# One-time setup: sync Python dependencies on a compute node.
#
# Usage: sbatch scripts/setup_venv.sh
#
# --- SLURM directives ---
#SBATCH -J setup_venv
#SBATCH -p cpu
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

echo "Syncing dependencies..."
uv sync --all-extras

echo "Done: uv sync"
