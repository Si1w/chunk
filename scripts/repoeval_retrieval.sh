#!/bin/bash
# Usage:
#   bash repoeval_retrieval.sh [config]                              # submit all (embed_model, split) pairs
#   bash repoeval_retrieval.sh <embed_model> <split> [config]        # submit single pair
#   sbatch ... repoeval_retrieval.sh --run <embed_model> <split> [config]  # execute (called by sbatch)
#
# --- SLURM directives ---
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

submit_job() {
    local embed_model="$1" split="$2" config="$3"
    local safe_name
    safe_name=$(echo "${embed_model}_${split}" | tr '/' '_')

    local JOB_ID
    if [ "${embed_model}" = "bm25" ]; then
        JOB_ID=$(sbatch \
            --job-name="repoeval_ret_${safe_name}" \
            --output="repoeval_ret_${safe_name}_%j.out" \
            --partition=cpu \
            "${SCRIPT_PATH}" --run "${embed_model}" "${split}" "${config}" \
            | awk '{print $4}')
    else
        JOB_ID=$(sbatch \
            --job-name="repoeval_ret_${safe_name}" \
            --output="repoeval_ret_${safe_name}_%j.out" \
            --partition=gpu \
            --gpus=1 \
            --constraint="a100|h100|h200|l40s" \
            "${SCRIPT_PATH}" --run "${embed_model}" "${split}" "${config}" \
            | awk '{print $4}')
    fi
    echo "Submitted: ${embed_model} / ${split} -> job ${JOB_ID}"
}

# --- Mode: execute (called by sbatch via --run flag) ---
if [ "${1:-}" = "--run" ]; then
    EMBED_MODEL="${2:?Missing embed_model}"
    SPLIT="${3:?Missing split}"
    CONFIG="${4:-${DEFAULT_CONFIG}}"

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

# --- Mode: submit single pair ---
if [ $# -ge 2 ] && [ "${1:-}" != "--run" ]; then
    EMBED_MODEL="$1"
    SPLIT="$2"
    CONFIG="${3:-${DEFAULT_CONFIG}}"
    submit_job "${EMBED_MODEL}" "${SPLIT}" "${CONFIG}"
    exit 0
fi

# --- Mode: submit all (embed_model, split) pairs ---
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
    submit_job "${embed_model}" "${split}" "${CONFIG}"
done <<< "${PAIRS}"
