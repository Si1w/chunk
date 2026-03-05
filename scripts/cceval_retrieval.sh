#!/bin/bash
# Usage:
#   bash cceval_retrieval.sh [config]                         # submit all embed_model jobs
#   bash cceval_retrieval.sh <embed_model> [config]           # submit single model
#   sbatch ... cceval_retrieval.sh --run <embed_model> [config]  # execute (called by sbatch)
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
DEFAULT_CONFIG="${PROJECT_DIR}/configs/cceval.yaml"

submit_job() {
    local embed_model="$1" config="$2"
    local safe_name
    safe_name=$(echo "${embed_model}" | tr '/' '_')

    local JOB_ID
    if [ "${embed_model}" = "bm25" ]; then
        JOB_ID=$(sbatch \
            --job-name="cceval_ret_${safe_name}" \
            --output="cceval_ret_${safe_name}_%j.out" \
            --partition=cpu \
            "${SCRIPT_PATH}" --run "${embed_model}" "${config}" \
            | awk '{print $4}')
    else
        JOB_ID=$(sbatch \
            --job-name="cceval_ret_${safe_name}" \
            --output="cceval_ret_${safe_name}_%j.out" \
            --partition=gpu \
            --gpus=1 \
            --constraint="a100|h100|h200|b200|l40s" \
            "${SCRIPT_PATH}" --run "${embed_model}" "${config}" \
            | awk '{print $4}')
    fi
    echo "Submitted: ${embed_model} -> job ${JOB_ID}"
}

# --- Mode: execute (called by sbatch via --run flag) ---
if [ "${1:-}" = "--run" ]; then
    EMBED_MODEL="${2:?Missing embed_model}"
    CONFIG="${3:-${DEFAULT_CONFIG}}"

    echo "=== Retrieval: ${EMBED_MODEL} ==="
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Start: $(date)"

    uv run python -m eval.cceval.retrieval \
        --config "${CONFIG}" \
        --embed_model "${EMBED_MODEL}"

    echo "=== Done: $(date) ==="
    exit 0
fi

# --- Mode: submit single model ---
if [ $# -ge 1 ] && [ "${1:-}" != "--run" ] && [[ "${1:-}" != *.yaml ]]; then
    EMBED_MODEL="$1"
    CONFIG="${2:-${DEFAULT_CONFIG}}"
    submit_job "${EMBED_MODEL}" "${CONFIG}"
    exit 0
fi

# --- Mode: submit all embed_model jobs ---
CONFIG="${1:-${DEFAULT_CONFIG}}"

MODELS=$(uv run python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
for m in cfg['retrieval']['embed_models']:
    if m == 'none':
        continue
    print(m)
")

while IFS= read -r embed_model; do
    submit_job "${embed_model}" "${CONFIG}"
done <<< "${MODELS}"
