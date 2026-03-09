#!/bin/bash
# Ablation study: sliding window overlap on RepoEval.
# Runs chunking + retrieval on CPU, then submits a GPU job for inference + scoring.
#
# Usage:
#   sbatch repoeval_ablation_overlap.sh [config]                # full pipeline
#   sbatch repoeval_ablation_overlap.sh [config] --skip_window  # skip chunking step
#
# --- SLURM directives (GPU for vLLM inference) ---
#SBATCH -J repoeval_ablation_overlap
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
CONFIG="${1:-${PROJECT_DIR}/configs/ablation_overlap.yaml}"

# Collect optional --skip_* flags
SKIP_FLAGS=""
for arg in "$@"; do
    case "$arg" in
        --skip_window|--skip_index|--skip_retrieval|--skip_completion)
            SKIP_FLAGS="${SKIP_FLAGS} ${arg}" ;;
    esac
done

echo "=== Overlap Ablation Study ==="
echo "Config: ${CONFIG}"
echo "Skip flags: ${SKIP_FLAGS:-none}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

uv run python -m eval.repoeval.ablation_overlap \
    --config "${CONFIG}" \
    ${SKIP_FLAGS}

echo "=== Ablation Done: $(date) ==="
