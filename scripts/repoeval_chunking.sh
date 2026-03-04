#!/bin/bash
# --- Job ---
#SBATCH -J chunking
#SBATCH -p cpu
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# --- Resources ---
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G

# --- Notifications ---
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_USER}@example.com

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${PROJECT_DIR}/configs/repoeval.yaml"

echo "=== Chunking ==="
echo "Config: ${CONFIG}"
echo "Start: $(date)"

uv run python -m eval.repoeval.make_window --config "${CONFIG}"

echo "=== Done: $(date) ==="
