#!/bin/bash
# Usage:
#   bash repoeval_retrieval.sh [config]                  # submit one job per embed_model
#   sbatch repoeval_retrieval.sh <embed_model>           # run single model (called by SLURM)
#
# --- SLURM directives (used when called via sbatch) ---
#SBATCH -J repoeval_ret
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -C "a100|h100|h200|b200|l40s"

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"
SCRIPT_PATH="$(realpath "$0")"
DEFAULT_CONFIG="${PROJECT_DIR}/configs/repoeval.yaml"

# --- Mode: run single model (called by sbatch) ---
if [ -n "${SLURM_JOB_ID:-}" ]; then
    EMBED_MODEL="${1:?Missing embed_model}"
    CONFIG="${2:-${DEFAULT_CONFIG}}"

    echo "=== Retrieval: ${EMBED_MODEL} ==="
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Start: $(date)"

    uv run python -m eval.repoeval.retrieval \
        --config "${CONFIG}" \
        --embed_model "${EMBED_MODEL}"

    echo "=== Done: $(date) ==="
    exit 0
fi

# --- Mode: submit all models (called by bash) ---
CONFIG="${1:-${DEFAULT_CONFIG}}"

EMBED_MODELS=$(uv run python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
for m in cfg['retrieval']['embed_models']:
    print(m)
")

while IFS= read -r embed_model; do
    if [ "${embed_model}" = "none" ]; then
        echo "Skipped: none (no retrieval baseline)"
        continue
    fi
    safe_name=$(echo "${embed_model}" | tr '/' '_')
    if [ "${embed_model}" = "bm25" ]; then
        JOB_ID=$(sbatch \
            --job-name="ret_${safe_name}" \
            --partition=cpu \
            --gres="" \
            --constraint="" \
            "${SCRIPT_PATH}" \
            "${embed_model}" "${CONFIG}" \
            | awk '{print $4}')
    else
        JOB_ID=$(sbatch \
            --job-name="ret_${safe_name}" \
            "${SCRIPT_PATH}" \
            "${embed_model}" "${CONFIG}" \
            | awk '{print $4}')
    fi
    echo "Submitted: ${embed_model} -> job ${JOB_ID}"
done <<< "${EMBED_MODELS}"
