#!/bin/bash
# Usage:
#   bash cceval_inference.sh [config]                                    # submit all (embed_model, llm) pairs
#   bash cceval_inference.sh <embed_model> <llm> [config]               # submit single pair
#   sbatch ... cceval_inference.sh --run <embed_model> <llm> [config]   # execute (called by sbatch)
#
# --- SLURM directives ---
#SBATCH -p gpu
#SBATCH -t 24:00:00
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
SCRIPT_PATH="$(realpath "$0")"
DEFAULT_CONFIG="${PROJECT_DIR}/configs/cceval.yaml"

submit_job() {
    local embed_model="$1" llm="$2" config="$3"
    local safe_name
    safe_name=$(echo "${embed_model}_${llm}" | tr '/' '_')

    local JOB_ID
    JOB_ID=$(sbatch \
        --job-name="cceval_inf_${safe_name}" \
        --output="cceval_inf_${safe_name}_%j.out" \
        "${SCRIPT_PATH}" --run "${embed_model}" "${llm}" "${config}" \
        | awk '{print $4}')
    echo "Submitted: ${embed_model} / ${llm} -> job ${JOB_ID}"
}

# --- Mode: execute (called by sbatch via --run flag) ---
if [ "${1:-}" = "--run" ]; then
    EMBED_MODEL="${2:?Missing embed_model}"
    LLM="${3:?Missing llm}"
    CONFIG="${4:-${DEFAULT_CONFIG}}"

    echo "=== Inference: ${EMBED_MODEL} / ${LLM} ==="
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Start: $(date)"

    uv run python -m eval.cceval.code_completion \
        --config "${CONFIG}" \
        --embed_model "${EMBED_MODEL}" \
        --llm "${LLM}"

    echo "=== Done: $(date) ==="
    exit 0
fi

# --- Mode: submit single pair ---
if [ $# -ge 2 ] && [ "${1:-}" != "--run" ]; then
    EMBED_MODEL="$1"
    LLM="$2"
    CONFIG="${3:-${DEFAULT_CONFIG}}"
    submit_job "${EMBED_MODEL}" "${LLM}" "${CONFIG}"
    exit 0
fi

# --- Mode: submit all (embed_model, llm) pairs ---
CONFIG="${1:-${DEFAULT_CONFIG}}"

PAIRS=$(uv run python -c "
import os, yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

chunking = cfg['chunking']
methods = ['cast', 'function', 'declaration', 'sliding'] if chunking['method'] == 'all' else [chunking['method']]
base_dir = os.path.join('${PROJECT_DIR}', 'eval', 'cceval', 'completion')
top_k = cfg['retrieval']['top_k']

def safe_name(name):
    return name.split('/')[-1]

def all_outputs_exist(embed_model, llm):
    if embed_model == 'none':
        path = os.path.join(base_dir, 'none', safe_name(llm), 'baseline_0_0_0.jsonl')
        return os.path.exists(path)
    for mcs in chunking['max_chunk_sizes']:
        for method in methods:
            for mcc in cfg['inference']['max_crossfile_contexts']:
                path = os.path.join(base_dir, safe_name(embed_model), safe_name(llm),
                    f'{method}_{mcs}_{mcc}_{top_k}.jsonl')
                if not os.path.exists(path):
                    return False
    return True

for m in cfg['retrieval']['embed_models']:
    for l in cfg['inference']['llms']:
        if not all_outputs_exist(m, l):
            print(f'{m}\t{l}')
        else:
            print(f'Skipped (exists): {m} / {l}', flush=True, file=__import__('sys').stderr)
")

if [ -n "${PAIRS}" ]; then
    while IFS=$'\t' read -r embed_model llm; do
        submit_job "${embed_model}" "${llm}" "${CONFIG}"
    done <<< "${PAIRS}"
else
    echo "All combinations already exist, nothing to submit."
fi
