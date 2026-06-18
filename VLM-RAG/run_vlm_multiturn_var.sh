#!/bin/bash
# =============================================================================
# run_vlm_multiturn_var.sh — SLURM launcher for Multi-Turn VAR evaluation
#
# GPU 1 experiment: structured cross-view reasoning on Test split (301 clips).
# One view per conversation turn, true history accumulation between turns.
# Evaluates BA_act and BA_sev — thesis contribution experiment.
#
# Usage:
#   sbatch run_vlm_multiturn_var.sh [max_samples] [model_key]
#
# Examples:
#   # Full test-set run (default, 4–6 h)
#   sbatch run_vlm_multiturn_var.sh
#
#   # Smoke test with 20 samples
#   sbatch run_vlm_multiturn_var.sh 20
#
#   # Different model (e.g. dense 30B)
#   sbatch run_vlm_multiturn_var.sh "" qwen3-vl-30b
# =============================================================================
#SBATCH --job-name=vlm_multiturn_var
#SBATCH --output=slurm_logs/vlm_multiturn_var_%j.out
#SBATCH --error=slurm_logs/vlm_multiturn_var_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist=h86
# NOTE: Qwen3-VL-30B-A3B weights ~60 GB (bf16) — fits on one 96 GB GPU.

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
MAX_SAMPLES="${1:-}"
MODEL_KEY="${2:-qwen3-vl-30b-a3b}"

# ── Paths ─────────────────────────────────────────────────────────────────────
VLM_RAG_DIR="${HOME}/thesis_automatic_faul_recognition/VLM-RAG"
VENV_DIR="${HOME}/thesis_automatic_faul_recognition/VARS_early_fusion/.venv"
# Annotations (JSON) live in SoccerNet_Data; compact frames in SoccerNet_HDF5_compact.
DATA_DIR="${HOME}/data/SoccerNet_Data"
HDF5_DIR="${HOME}/data/SoccerNet_HDF5_compact"
HF_CACHE="${HOME}/.cache/huggingface"
RESULTS_DIR="${VLM_RAG_DIR}/results"

mkdir -p "${RESULTS_DIR}" \
         "${HOME}/thesis_automatic_faul_recognition/VLM-RAG/slurm_logs"

echo "=== Multi-Turn VAR Evaluation ==="
echo "Node      : $(hostname)"
echo "GPUs      : ${CUDA_VISIBLE_DEVICES:-<SLURM-assigned>}"
echo "Model     : ${MODEL_KEY}"
echo "Max samp  : ${MAX_SAMPLES:-all}"
echo "Start     : $(date)"

# ── Environment ───────────────────────────────────────────────────────────────
# Use the VARS_early_fusion venv (same deps: transformers, qwen_vl_utils, h5py)
module load uv
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Building venv via uv (first run)..."
    cd "${HOME}/thesis_automatic_faul_recognition/VARS_early_fusion"
    uv sync --reinstall
fi

source "${VENV_DIR}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${VLM_RAG_DIR}"

# ── Build argument list ────────────────────────────────────────────────────────
ARGS=(
    --vlm_model     "${MODEL_KEY}"
    --data_dir      "${DATA_DIR}"
    --hdf5_dir      "${HDF5_DIR}"
    --split         Test
    --thinking
    --num_frames    8
    --max_views     4
    --tokens_per_view 768
    --tokens_final    512
    --output_path   "${RESULTS_DIR}/multiturn_var_${MODEL_KEY}_test.json"
    --hf_cache_dir  "${HF_CACHE}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
    ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

# ── Run ───────────────────────────────────────────────────────────────────────
python vlm_multiturn_var.py "${ARGS[@]}"

echo ""
echo "=== Done: $(date) ==="
echo "Results → ${RESULTS_DIR}/multiturn_var_${MODEL_KEY}_test.json"
