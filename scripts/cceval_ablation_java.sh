#!/bin/bash
# Cross-language validation: cceval on Java
#
# Usage:
#   bash scripts/cceval_ablation_java.sh [config]                                          # submit full pipeline
#   bash scripts/cceval_ablation_java.sh --step <step> [--embed_model M|--llm L] [config]  # submit single step
#   sbatch ... scripts/cceval_ablation_java.sh --run <step> [--embed_model M|--llm L] [config]  # execute
#
# --- SLURM directives ---
#SBATCH -t 24:00:00
#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
SCRIPT_PATH="$(realpath "$0")"
DEFAULT_CONFIG="${PROJECT_DIR}/configs/ablation_java.yaml"

# --- Execute mode (called by sbatch via --run flag) ---
if [ "${1:-}" = "--run" ]; then
    STEP="${2:?Missing step}"
    shift 2

    EXTRA_ARGS=()
    CONFIG="${DEFAULT_CONFIG}"
    while [ $# -gt 0 ]; do
        case "$1" in
            --embed_model) EXTRA_ARGS+=(--embed_model "$2"); shift 2 ;;
            --llm)         EXTRA_ARGS+=(--llm "$2"); shift 2 ;;
            *)             CONFIG="$1"; shift ;;
        esac
    done

    echo "=== cceval Java: ${STEP} ==="
    echo "Config: ${CONFIG}"
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Start: $(date)"

    uv run python -m eval.cceval.ablation_java \
        --config "${CONFIG}" \
        --steps "${STEP}" \
        "${EXTRA_ARGS[@]}"

    echo "=== Done: $(date) ==="
    exit 0
fi

# --- Submit single step ---
if [ "${1:-}" = "--step" ]; then
    STEP="${2:?Missing step}"
    shift 2

    EXTRA_ARGS=()
    CONFIG="${DEFAULT_CONFIG}"
    while [ $# -gt 0 ]; do
        case "$1" in
            --embed_model) EXTRA_ARGS+=(--embed_model "$2"); shift 2 ;;
            --llm)         EXTRA_ARGS+=(--llm "$2"); shift 2 ;;
            *)             CONFIG="$1"; shift ;;
        esac
    done

    case "${STEP}" in
        fetch|chunk|score)
            JOB_ID=$(sbatch \
                --job-name="cceval_java_${STEP}" \
                --partition=cpu \
                --mem=32G \
                "${SCRIPT_PATH}" --run "${STEP}" "${EXTRA_ARGS[@]}" "${CONFIG}" \
                | awk '{print $4}')
            echo "Submitted ${STEP} -> job ${JOB_ID}"
            ;;
        retrieve)
            if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
                em="${EXTRA_ARGS[1]}"
                safe_name=$(echo "cceval_java_ret_${em}" | tr '/' '_')
                if [ "${em}" = "bm25" ]; then
                    JOB_ID=$(sbatch \
                        --job-name="${safe_name}" \
                        --partition=cpu \
                        "${SCRIPT_PATH}" --run retrieve --embed_model "${em}" "${CONFIG}" \
                        | awk '{print $4}')
                else
                    JOB_ID=$(sbatch \
                        --job-name="${safe_name}" \
                        --partition=gpu \
                        --gpus=1 \
                        --constraint="a100|h100|h200|l40s" \
                        "${SCRIPT_PATH}" --run retrieve --embed_model "${em}" "${CONFIG}" \
                        | awk '{print $4}')
                fi
                echo "Submitted retrieve ${em} -> job ${JOB_ID}"
            else
                EMBED_MODELS=$(uv run python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
for m in cfg['retrieval']['embed_models']:
    if m != 'none':
        print(m)
")
                while IFS= read -r em; do
                    safe_name=$(echo "cceval_java_ret_${em}" | tr '/' '_')
                    if [ "${em}" = "bm25" ]; then
                        JOB_ID=$(sbatch \
                            --job-name="${safe_name}" \
                            --partition=cpu \
                            "${SCRIPT_PATH}" --run retrieve --embed_model "${em}" "${CONFIG}" \
                            | awk '{print $4}')
                    else
                        JOB_ID=$(sbatch \
                            --job-name="${safe_name}" \
                            --partition=gpu \
                            --gpus=1 \
                            --constraint="a100|h100|h200|l40s" \
                            "${SCRIPT_PATH}" --run retrieve --embed_model "${em}" "${CONFIG}" \
                            | awk '{print $4}')
                    fi
                    echo "Submitted retrieve ${em} -> job ${JOB_ID}"
                done <<< "${EMBED_MODELS}"
            fi
            ;;
        infer)
            if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
                llm="${EXTRA_ARGS[1]}"
                safe_name=$(echo "cceval_java_infer_${llm}" | tr '/' '_')
                JOB_ID=$(sbatch \
                    --job-name="${safe_name}" \
                    --partition=gpu \
                    --gpus=1 \
                    --constraint="a100|h100|h200|l40s" \
                    "${SCRIPT_PATH}" --run infer --llm "${llm}" "${CONFIG}" \
                    | awk '{print $4}')
                echo "Submitted infer ${llm} -> job ${JOB_ID}"
            else
                LLMS=$(uv run python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
for m in cfg['inference']['llms']:
    print(m)
")
                while IFS= read -r llm; do
                    safe_name=$(echo "cceval_java_infer_${llm}" | tr '/' '_')
                    JOB_ID=$(sbatch \
                        --job-name="${safe_name}" \
                        --partition=gpu \
                        --gpus=1 \
                        --constraint="a100|h100|h200|l40s" \
                        "${SCRIPT_PATH}" --run infer --llm "${llm}" "${CONFIG}" \
                        | awk '{print $4}')
                    echo "Submitted infer ${llm} -> job ${JOB_ID}"
                done <<< "${LLMS}"
            fi
            ;;
    esac
    exit 0
fi

# --- Submit full pipeline with dependencies ---
CONFIG="${1:-${DEFAULT_CONFIG}}"

# Step 0: fetch (CPU)
JOB_FETCH=$(sbatch \
    --job-name="cceval_java_fetch" \
    --partition=cpu \
    --mem=16G \
    "${SCRIPT_PATH}" --run fetch "${CONFIG}" \
    | awk '{print $4}')
echo "Submitted fetch -> job ${JOB_FETCH}"

# Step 1: chunk (CPU, depends on fetch)
JOB_CHUNK=$(sbatch \
    --job-name="cceval_java_chunk" \
    --dependency="afterok:${JOB_FETCH}" \
    --partition=cpu \
    --mem=32G \
    "${SCRIPT_PATH}" --run chunk "${CONFIG}" \
    | awk '{print $4}')
echo "Submitted chunk -> job ${JOB_CHUNK}"

# Step 2: retrieve — one job per embed_model, depends on chunk
EMBED_MODELS=$(uv run python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
for m in cfg['retrieval']['embed_models']:
    if m != 'none':
        print(m)
")

RETRIEVE_JOBS=""
while IFS= read -r em; do
    safe_name=$(echo "${em}" | tr '/' '_')
    if [ "${em}" = "bm25" ]; then
        JOB_ID=$(sbatch \
            --job-name="cceval_java_ret_${safe_name}" \
            --dependency="afterok:${JOB_CHUNK}" \
            --partition=cpu \
            "${SCRIPT_PATH}" --run retrieve --embed_model "${em}" "${CONFIG}" \
            | awk '{print $4}')
    else
        JOB_ID=$(sbatch \
            --job-name="cceval_java_ret_${safe_name}" \
            --dependency="afterok:${JOB_CHUNK}" \
            --partition=gpu \
            --gpus=1 \
            --constraint="a100|h100|h200|l40s" \
            "${SCRIPT_PATH}" --run retrieve --embed_model "${em}" "${CONFIG}" \
            | awk '{print $4}')
    fi
    echo "Submitted retrieve ${em} -> job ${JOB_ID}"
    RETRIEVE_JOBS="${RETRIEVE_JOBS}:${JOB_ID}"
done <<< "${EMBED_MODELS}"

# Step 3: infer — one job per LLM (model loaded once, reused across all combos)
LLMS=$(uv run python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
for m in cfg['inference']['llms']:
    print(m)
")

INFER_JOBS=""
while IFS= read -r llm; do
    safe_name=$(echo "${llm}" | tr '/' '_')
    JOB_ID=$(sbatch \
        --job-name="cceval_java_infer_${safe_name}" \
        --dependency="afterok${RETRIEVE_JOBS}" \
        --partition=gpu \
        --gpus=1 \
        --constraint="a100|h100|h200|l40s" \
        "${SCRIPT_PATH}" --run infer --llm "${llm}" "${CONFIG}" \
        | awk '{print $4}')
    echo "Submitted infer ${llm} -> job ${JOB_ID}"
    INFER_JOBS="${INFER_JOBS}:${JOB_ID}"
done <<< "${LLMS}"

# Step 4: score (CPU, depends on all infer jobs)
JOB_SCORE=$(sbatch \
    --job-name="cceval_java_score" \
    --dependency="afterok${INFER_JOBS}" \
    --partition=cpu \
    --mem=16G \
    "${SCRIPT_PATH}" --run score "${CONFIG}" \
    | awk '{print $4}')
echo "Submitted score -> job ${JOB_SCORE}"

echo ""
echo "Pipeline: fetch(${JOB_FETCH}) -> chunk(${JOB_CHUNK}) -> retrieve -> infer -> score(${JOB_SCORE})"
