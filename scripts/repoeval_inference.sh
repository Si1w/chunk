#!/bin/bash
# Usage:
#   bash repoeval_inference.sh [config]                              # submit all (embed_model, split, llm) jobs
#   sbatch repoeval_inference.sh <embed_model> <split> <llm>         # run single triple (called by SLURM)
#
# --- SLURM directives (used when called via sbatch) ---
#SBATCH -J repoeval_inf
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -o %x_%j.out
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

# --- Mode: run single (embed_model, split, llm) triple (called by sbatch) ---
if [ -n "${SLURM_JOB_ID:-}" ]; then
    EMBED_MODEL="${1:?Missing embed_model}"
    SPLIT="${2:?Missing split}"
    LLM="${3:?Missing llm}"
    CONFIG="${4:-${DEFAULT_CONFIG}}"

    echo "=== Inference: ${EMBED_MODEL} / ${SPLIT} / ${LLM} ==="
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Start: $(date)"

    uv run python -m eval.repoeval.code_completion \
        --config "${CONFIG}" \
        --embed_model "${EMBED_MODEL}" \
        --split "${SPLIT}" \
        --llm "${LLM}"

    echo "=== Done: $(date) ==="
    exit 0
fi

# --- Mode: submit all (embed_model, split, llm) triples (called by bash) ---
CONFIG="${1:-${DEFAULT_CONFIG}}"

TRIPLES=$(uv run python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
split = cfg.get('evaluation', {}).get('split', 'both')
splits = ['api', 'line'] if split == 'both' else [split]
for m in cfg['retrieval']['embed_models']:
    for s in splits:
        for l in cfg['inference']['llms']:
            print(f'{m}\t{s}\t{l}')
")

while IFS=$'\t' read -r embed_model split llm; do
    safe_name=$(echo "${embed_model}_${split}_${llm}" | tr '/' '_')
    JOB_ID=$(sbatch \
        --job-name="inf_${safe_name}" \
        "${SCRIPT_PATH}" \
        "${embed_model}" "${split}" "${llm}" "${CONFIG}" \
        | awk '{print $4}')
    echo "Submitted: ${embed_model} / ${split} / ${llm} -> job ${JOB_ID}"
done <<< "${TRIPLES}"
