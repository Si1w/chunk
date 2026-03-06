#!/bin/bash
# Usage:
#   bash repoeval_inference.sh [config]                                        # submit all (embed_model, split, llm) triples
#   bash repoeval_inference.sh <embed_model> <split> <llm> [config]            # submit single triple
#   sbatch ... repoeval_inference.sh --run <embed_model> <split> <llm> [config] # execute (called by sbatch)
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
#SBATCH -C "a100|h100|h200|b200|l40s"

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
SCRIPT_PATH="$(realpath "$0")"
DEFAULT_CONFIG="${PROJECT_DIR}/configs/repoeval.yaml"

submit_job() {
    local embed_model="$1" split="$2" llm="$3" config="$4"
    local safe_name
    safe_name=$(echo "${embed_model}_${split}_${llm}" | tr '/' '_')

    local JOB_ID
    JOB_ID=$(sbatch \
        --job-name="repoeval_inf_${safe_name}" \
        --output="repoeval_inf_${safe_name}_%j.out" \
        "${SCRIPT_PATH}" --run "${embed_model}" "${split}" "${llm}" "${config}" \
        | awk '{print $4}')
    echo "Submitted: ${embed_model} / ${split} / ${llm} -> job ${JOB_ID}"
}

# --- Mode: execute (called by sbatch via --run flag) ---
if [ "${1:-}" = "--run" ]; then
    EMBED_MODEL="${2:?Missing embed_model}"
    SPLIT="${3:?Missing split}"
    LLM="${4:?Missing llm}"
    CONFIG="${5:-${DEFAULT_CONFIG}}"

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

# --- Mode: submit single triple ---
if [ $# -ge 3 ] && [ "${1:-}" != "--run" ]; then
    EMBED_MODEL="$1"
    SPLIT="$2"
    LLM="$3"
    CONFIG="${4:-${DEFAULT_CONFIG}}"
    submit_job "${EMBED_MODEL}" "${SPLIT}" "${LLM}" "${CONFIG}"
    exit 0
fi

# --- Mode: submit all (embed_model, split, llm) triples ---
CONFIG="${1:-${DEFAULT_CONFIG}}"

TRIPLES=$(uv run python -c "
import os, yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

split = cfg.get('evaluation', {}).get('split', 'both')
splits = ['api', 'line'] if split == 'both' else [split]
chunking = cfg['chunking']
methods = ['cast', 'function', 'declaration', 'sliding'] if chunking['method'] == 'all' else [chunking['method']]
base_dir = os.path.join('${PROJECT_DIR}', 'eval', 'repoeval', 'completion')
top_k = cfg['retrieval']['top_k']

def safe_name(name):
    return name.split('/')[-1]

def all_outputs_exist(embed_model, s, llm):
    if embed_model == 'none':
        path = os.path.join(base_dir, 'none', safe_name(llm), f'{s}_baseline_0_0_0.jsonl')
        return os.path.exists(path)
    for mcs in chunking['max_chunk_sizes']:
        for method in methods:
            for mcc in cfg['inference']['max_crossfile_contexts']:
                path = os.path.join(base_dir, safe_name(embed_model), safe_name(llm),
                    f'{s}_{method}_{mcs}_{mcc}_{top_k}.jsonl')
                if not os.path.exists(path):
                    return False
    return True

for m in cfg['retrieval']['embed_models']:
    for s in splits:
        for l in cfg['inference']['llms']:
            if not all_outputs_exist(m, s, l):
                print(f'{m}\t{s}\t{l}')
            else:
                print(f'Skipped (exists): {m} / {s} / {l}', flush=True, file=__import__('sys').stderr)
")

while IFS=$'\t' read -r embed_model split llm; do
    submit_job "${embed_model}" "${split}" "${llm}" "${CONFIG}"
done <<< "${TRIPLES}"
