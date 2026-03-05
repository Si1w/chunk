#!/bin/bash
# Usage:
#   bash repoeval_retrieval.sh [config]                          # submit one job per (embed_model, split)
#   sbatch repoeval_retrieval.sh <embed_model> <split>           # run single pair (called by SLURM)
#
# --- SLURM directives (used when called via sbatch) ---
#SBATCH -J repoeval_ret
#SBATCH -t 12:00:00
#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
SCRIPT_PATH="$(realpath "$0")"
DEFAULT_CONFIG="${PROJECT_DIR}/configs/repoeval.yaml"

# --- Mode: run single (embed_model, split) pair (called by sbatch) ---
if [ -n "${SLURM_JOB_ID:-}" ]; then
    EMBED_MODEL="${1:?Missing embed_model}"
    SPLIT="${2:?Missing split}"
    CONFIG="${3:-${DEFAULT_CONFIG}}"

    echo "=== Retrieval: ${EMBED_MODEL} / ${SPLIT} ==="
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Start: $(date)"

    uv run python -m eval.repoeval.retrieval \
        --config "${CONFIG}" \
        --embed_model "${EMBED_MODEL}" \
        --split "${SPLIT}"

    echo "=== Done: $(date) ==="
    exit 0
fi

# --- Mode: submit all (embed_model, split) pairs (called by bash) ---
CONFIG="${1:-${DEFAULT_CONFIG}}"

PAIRS=$(uv run python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
split = cfg.get('evaluation', {}).get('split', 'both')
splits = ['api', 'line'] if split == 'both' else [split]
for m in cfg['retrieval']['embed_models']:
    if m == 'none':
        continue
    for s in splits:
        print(f'{m}\t{s}')
")

while IFS=$'\t' read -r embed_model split; do
    safe_name=$(echo "${embed_model}_${split}" | tr '/' '_')
    if [ "${embed_model}" = "bm25" ]; then
        JOB_ID=$(sbatch \
            --job-name="ret_${safe_name}" \
            --partition=cpu \
            "${SCRIPT_PATH}" \
            "${embed_model}" "${split}" "${CONFIG}" \
            | awk '{print $4}')
    else
        JOB_ID=$(sbatch \
            --job-name="ret_${safe_name}" \
            --partition=gpu \
            --gpus=1 \
            --constraint="a100|h100|h200|b200|l40s" \
            "${SCRIPT_PATH}" \
            "${embed_model}" "${split}" "${CONFIG}" \
            | awk '{print $4}')
    fi
    echo "Submitted: ${embed_model} / ${split} -> job ${JOB_ID}"
done <<< "${PAIRS}"
