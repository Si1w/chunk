#!/bin/bash
# Ablation study: sliding window overlap on RepoEval.
# Submits one GPU job per LLM so each job loads the model only once.
#
# Usage:
#   bash repoeval_ablation_overlap.sh [config]                                  # submit all LLMs
#   bash repoeval_ablation_overlap.sh <llm> [config]                            # submit single LLM
#   bash repoeval_ablation_overlap.sh [--skip_window] [--skip_retrieval] [--skip_completion] [config]
#   sbatch ... repoeval_ablation_overlap.sh --run <llm> [config] [--skip_*]     # execute (called by sbatch)
#
# --- SLURM directives (GPU for vLLM inference) ---
#SBATCH -p gpu
#SBATCH -t 48:00:00
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
DEFAULT_CONFIG="${PROJECT_DIR}/configs/ablation_overlap.yaml"

# --- Parse --skip_* flags from any position ---
SKIP_FLAGS=()
POSITIONAL=()
for arg in "$@"; do
    case "${arg}" in
        --skip_window|--skip_retrieval|--skip_completion)
            SKIP_FLAGS+=("${arg}") ;;
        *)
            POSITIONAL+=("${arg}") ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

submit_job() {
    local llm="$1" config="$2"
    local safe_name
    safe_name=$(echo "${llm}" | tr '/' '_')

    local JOB_ID
    JOB_ID=$(sbatch \
        --job-name="ablation_overlap_${safe_name}" \
        --output="ablation_overlap_${safe_name}_%j.out" \
        "${SCRIPT_PATH}" --run "${llm}" "${config}" "${SKIP_FLAGS[@]+"${SKIP_FLAGS[@]}"}" \
        | awk '{print $4}')
    echo "Submitted: ${llm} -> job ${JOB_ID}"
}

# --- Mode: execute (called by sbatch via --run flag) ---
if [ "${1:-}" = "--run" ]; then
    LLM="${2:?Missing llm}"
    CONFIG="${3:-${DEFAULT_CONFIG}}"

    echo "=== Overlap Ablation: ${LLM} ==="
    echo "Config: ${CONFIG}"
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Start: $(date)"

    uv run python -m eval.repoeval.ablation_overlap \
        --config "${CONFIG}" \
        --llm "${LLM}" \
        "${SKIP_FLAGS[@]+"${SKIP_FLAGS[@]}"}"

    echo "=== Done: $(date) ==="
    exit 0
fi

# --- Mode: submit single LLM ---
if [ $# -ge 1 ] && [ "${1:-}" != "--run" ] && [[ "${1:-}" == */* ]]; then
    LLM="$1"
    CONFIG="${2:-${DEFAULT_CONFIG}}"
    submit_job "${LLM}" "${CONFIG}"
    exit 0
fi

# --- Mode: submit all LLMs ---
CONFIG="${1:-${DEFAULT_CONFIG}}"

LLMS=$(uv run python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
for llm in cfg['inference']['llms']:
    print(llm)
")

while IFS= read -r llm; do
    submit_job "${llm}" "${CONFIG}"
done <<< "${LLMS}"
