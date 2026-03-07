#!/bin/bash
# Usage:
#   bash scripts/repoeval_score.sh [config]              # submit as SLURM job
#   sbatch scripts/repoeval_score.sh --run [config]      # execute (called by sbatch)
#
# --- SLURM directives ---
#SBATCH -J repoeval_score
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
    CONFIG="${2:-configs/repoeval.yaml}"

    echo "=== Compute Scores ==="
    echo "Config: ${CONFIG}"
    echo "Start: $(date)"

    uv run python -m eval.repoeval.compute_score --config "${CONFIG}"

    echo "=== Done: $(date) ==="
    exit 0
fi

# --- Mode: submit job ---
CONFIG="${1:-configs/repoeval.yaml}"

JOB_ID=$(sbatch \
    --job-name="repoeval_score" \
    --output="repoeval_score_%j.out" \
    "${SCRIPT_PATH}" --run "${CONFIG}" \
    | awk '{print $4}')
echo "Submitted: repoeval_score -> job ${JOB_ID}"
