#!/bin/bash
# Usage:
#   bash scripts/cceval_score.sh [config]              # submit as SLURM job
#   sbatch scripts/cceval_score.sh --run [config]      # execute (called by sbatch)
#
# --- SLURM directives ---
#SBATCH -J cceval_score
#SBATCH -p cpu
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
SCRIPT_PATH="$(realpath "$0")"

# --- Mode: execute (called by sbatch via --run flag) ---
if [ "${1:-}" = "--run" ]; then
    CONFIG="${2:-configs/cceval.yaml}"

    echo "=== Compute Scores ==="
    echo "Config: ${CONFIG}"
    echo "Start: $(date)"

    uv run python -m eval.cceval.compute_score --config "${CONFIG}"

    echo "=== Done: $(date) ==="
    exit 0
fi

# --- Mode: submit job ---
CONFIG="${1:-configs/cceval.yaml}"

JOB_ID=$(sbatch \
    --job-name="cceval_score" \
    --output="cceval_score_%j.out" \
    "${SCRIPT_PATH}" --run "${CONFIG}" \
    | awk '{print $4}')
echo "Submitted: cceval_score -> job ${JOB_ID}"
