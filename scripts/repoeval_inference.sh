#!/bin/bash
# Usage:
#   bash repoeval_inference.sh [config]                    # submit all (embed_model, llm) pairs
#   sbatch repoeval_inference.sh <embed_model> <llm>       # run single pair (called by SLURM)
#
# --- SLURM directives (used when called via sbatch) ---
#SBATCH -J repoeval_inf
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -C a100

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_PATH="$(realpath "$0")"
DEFAULT_CONFIG="${PROJECT_DIR}/configs/repoeval.yaml"

# --- Mode: run single pair (called by sbatch) ---
if [ -n "${SLURM_JOB_ID:-}" ]; then
    EMBED_MODEL="${1:?Missing embed_model}"
    LLM="${2:?Missing llm}"
    CONFIG="${3:-${DEFAULT_CONFIG}}"

    echo "=== Inference: ${EMBED_MODEL} + ${LLM} ==="
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Start: $(date)"

    cd "${PROJECT_DIR}"
    uv run python -m eval.repoeval.code_completion \
        --config "${CONFIG}" \
        --embed_model "${EMBED_MODEL}" \
        --llm "${LLM}"

    echo "=== Done: $(date) ==="
    exit 0
fi

# --- Mode: submit all pairs (called by bash) ---
CONFIG="${1:-${DEFAULT_CONFIG}}"

PAIRS=$(cd "${PROJECT_DIR}" && uv run python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
for m in cfg['retrieval']['embed_models']:
    for l in cfg['inference']['llms']:
        print(f'{m}\t{l}')
")

JOB_IDS=""
while IFS=$'\t' read -r embed_model llm; do
    safe_name=$(echo "${embed_model}_${llm}" | tr '/' '_')
    JOB_ID=$(sbatch \
        --job-name="inf_${safe_name}" \
        "${SCRIPT_PATH}" \
        "${embed_model}" "${llm}" "${CONFIG}" \
        | awk '{print $4}')
    echo "Submitted: ${embed_model} + ${llm} -> job ${JOB_ID}"
    JOB_IDS="${JOB_IDS}:${JOB_ID}"
done <<< "${PAIRS}"

# Compute scores after all inference jobs finish
SCORE_ID=$(sbatch \
    --dependency=afterok${JOB_IDS} \
    --job-name="repoeval_score" \
    --partition=cpu \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --output=%x_%j.out \
    --error=%x_%j.err \
    --wrap="cd ${PROJECT_DIR} && uv run python -m eval.repoeval.compute_score --config ${CONFIG}" \
    | awk '{print $4}')

echo ""
echo "Score job ${SCORE_ID} will run after all inference completes."
echo "You can disconnect now."
