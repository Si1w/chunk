#!/bin/bash
# Pilot experiment: run the full pipeline with limited queries to verify code.
#
# Usage: sbatch cceval_pilot.sh [config]
#
#SBATCH -J cceval_pilot
#SBATCH -p gpu
#SBATCH -t 02:00:00
#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -C "a100|h100|h200|l40s"

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
CONFIG="${1:-${PROJECT_DIR}/configs/cceval_pilot.yaml}"

NUM_QUERIES=$(uv run python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('pilot', {}).get('num_queries', 5))
")

echo "=== Pilot Experiment (cceval) ==="
echo "Config: ${CONFIG}"
echo "Queries: ${NUM_QUERIES}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# Step 0: Fetch Dataset
echo "--- Step 0: Fetch Dataset ---"
uv run python -m eval.cceval.fetch_dataset

# Step 1: Chunking
echo "--- Step 1: Chunking ---"
uv run python -m eval.cceval.make_window --config "${CONFIG}"

# Step 2: Retrieval
echo "--- Step 2: Retrieval ---"
uv run python -m eval.cceval.retrieval --config "${CONFIG}" --num_queries "${NUM_QUERIES}"

# Step 3: Code Completion
echo "--- Step 3: Code Completion ---"
uv run python -m eval.cceval.code_completion --config "${CONFIG}"

# Step 4: Compute Scores
echo "--- Step 4: Compute Scores ---"
uv run python -m eval.cceval.compute_score --config "${CONFIG}"

echo "=== Pilot Done: $(date) ==="
