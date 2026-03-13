#!/bin/bash
# Ablation study: sliding window overlap on RepoEval.
# Submits one GPU job per LLM so each job loads the model only once.
#
# Usage:
#   bash repoeval_ablation_overlap.sh --steps <steps> [config]           # submit all LLMs
#   bash repoeval_ablation_overlap.sh --steps <steps> <llm> [config]     # submit single LLM
#   Steps: window, retrieval, completion, score
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

# --- Parse all flags and positional args ---
RUN_MODE=false
STEPS=()
POSITIONAL=()
PARSING_STEPS=false
for arg in "$@"; do
    if [ "${arg}" = "--run" ]; then
        RUN_MODE=true
        continue
    fi
    if [ "${arg}" = "--steps" ]; then
        PARSING_STEPS=true
        continue
    fi
    if [ "${PARSING_STEPS}" = true ]; then
        case "${arg}" in
            window|retrieval|completion|score)
                STEPS+=("${arg}")
                ;;
            *)
                PARSING_STEPS=false
                POSITIONAL+=("${arg}")
                ;;
        esac
    else
        POSITIONAL+=("${arg}")
    fi
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

if [ ${#STEPS[@]} -eq 0 ]; then
    echo "Error: --steps is required (window, retrieval, completion, score)" >&2
    exit 1
fi
STEPS_STR="${STEPS[*]}"

# --- Mode: execute (called by sbatch via --run flag) ---
if [ "${RUN_MODE}" = true ]; then
    LLM="${1:?Missing llm}"
    CONFIG="${2:-${DEFAULT_CONFIG}}"

    echo "=== Overlap Ablation: ${LLM} ==="
    echo "Config: ${CONFIG}"
    echo "Steps: ${STEPS_STR}"
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Start: $(date)"

    # shellcheck disable=SC2086
    uv run python -m eval.repoeval.ablation_overlap \
        --config "${CONFIG}" \
        --llm "${LLM}" \
        --steps ${STEPS_STR}

    echo "=== Done: $(date) ==="
    exit 0
fi

# --- Submit mode ---
submit_job() {
    local llm="$1" config="$2"
    local safe_name
    safe_name=$(echo "${llm}" | tr '/' '_')

    local JOB_ID
    # shellcheck disable=SC2086
    JOB_ID=$(sbatch \
        --job-name="ablation_overlap_${safe_name}" \
        --output="ablation_overlap_${safe_name}_%j.out" \
        "${SCRIPT_PATH}" --run --steps ${STEPS_STR} "${llm}" "${config}" \
        | awk '{print $4}')
    echo "Submitted: ${llm} -> job ${JOB_ID}"
}

# Submit single LLM
if [ $# -ge 1 ] && [[ "${1:-}" == */* ]]; then
    LLM="$1"
    CONFIG="${2:-${DEFAULT_CONFIG}}"
    submit_job "${LLM}" "${CONFIG}"
    exit 0
fi

# Submit all LLMs from config
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
