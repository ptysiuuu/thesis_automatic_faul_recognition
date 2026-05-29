#!/bin/bash
# =============================================================================
# run_multi_turn_var.sh — SLURM launcher for the Multi-Turn VAR strategy
#
# Usage:
#   sbatch run_multi_turn_var.sh [max_samples] [model_name] [output_subdir] [split]
#
# Positional arguments (all optional):
#   $1  max_samples   — integer cap on evaluation samples (omit for full run)
#   $2  model_name    — HuggingFace / NVIDIA / Gemini model identifier
#   $3  output_subdir — subfolder inside ablation_results_v2/ (default: multi_turn_var)
#   $4  split         — valid | test | chall  (default: valid)
#
# Examples:
#   # Smoke test — 50 samples, Qwen 7B
#   sbatch run_multi_turn_var.sh 50
#
#   # Full valid-set run
#   sbatch run_multi_turn_var.sh
#
#   # With 32B model (requires more GPU memory — consider 2×A100 if OOM)
#   sbatch run_multi_turn_var.sh 50 Qwen/Qwen2.5-VL-32B-Instruct multi_turn_32b
#
#   # NVIDIA NIM (API-based, no GPU memory needed for model weights)
#   sbatch run_multi_turn_var.sh 50 nvidia:mistralai/mistral-large-3-675b-instruct-2512 multi_turn_nim
#
#   # Test split
#   sbatch run_multi_turn_var.sh "" Qwen/Qwen2.5-VL-7B-Instruct multi_turn_test test
#
# Strategy details:
#   Turn 1 (5 frames): 2 live + 3 replay → physical description only
#   Turn 2 (4 frames): dense contact cluster → forced-coverage checklist
#   Turn 3 (4 frames): aftermath-biased → ordinal IFAB severity cascade
#   frames_per_view=8 with weighted sampling (centre-biased) for best coverage.
#
# Outputs saved to:  ablation_results_v2/<output_subdir>/results.json
#                                                          summary.json
# =============================================================================
#SBATCH --job-name=vlm_multi_turn_var
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=multi_turn_var_%x_%j.out

set -euo pipefail

export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME"

# ── Parse arguments ───────────────────────────────────────────────────────────
MAX_SAMPLES="${1:-}"
MODEL_NAME="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_SUBDIR="${3:-multi_turn_var}"
SPLIT="${4:-valid}"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT="/net/tscratch/people/plgaszos/SoccerNet_Data"
HDF5_ROOT="/net/tscratch/people/plgaszos/SoccerNet_HDF5"
FAISS_DIR="/net/tscratch/people/plgaszos/vlm_rag_icl"
SCRIPT_DIR="/net/tscratch/people/plgaszos/sn-mvfoul/VLM-RAG/rag_icl"
OUTPUT_DIR="ablation_results_v2/${OUTPUT_SUBDIR}"

# ── Resolve split-specific file paths ────────────────────────────────────────
split_lc=$(echo "$SPLIT" | tr '[:upper:]' '[:lower:]')
case "$split_lc" in
    train)          split_cap="Train" ;;
    valid|val)      split_cap="Valid" ;;
    test)           split_cap="Test"  ;;
    chall|challenge) split_cap="Chall" ;;
    *)              split_cap="Valid" ;;
esac

HDF5_PATH="$HDF5_ROOT/${split_cap}.hdf5"
ANNOTATIONS="$DATA_ROOT/${split_cap}/annotations.json"

# ── Activate conda environment ────────────────────────────────────────────────
source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm32b
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Strategy:    multi_turn_var"
echo "Model:       $MODEL_NAME"
echo "Split:       $SPLIT  ($HDF5_PATH)"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Output:      $OUTPUT_DIR"
echo "frames/view: 8  (weighted, centre-biased)"
echo "Turns:       3  (description → checklist → severity cascade)"
echo "============================================================"

# ── Build argument array ──────────────────────────────────────────────────────
ARGS=(
    --hdf5_path         "$HDF5_PATH"
    --annotations       "$ANNOTATIONS"
    --train_annotations "$DATA_ROOT/Train/annotations.json"
    --law12_pdf         "$DATA_ROOT/law12.pdf"
    --faiss_index_path  "$FAISS_DIR/train_mvit_features.index"
    --faiss_meta_path   "$FAISS_DIR/train_mvit_metadata.json"
    --medoid_cache      "$FAISS_DIR/medoid_cache.json"
    --strategy          "multi_turn_var"
    --model_name        "$MODEL_NAME"
    --frames_per_view   8
    --retrieval_k       3
    --output_dir        "$OUTPUT_DIR"
)

if [ -n "$MAX_SAMPLES" ]; then
    ARGS+=(--max_samples "$MAX_SAMPLES")
fi

# ── Run ───────────────────────────────────────────────────────────────────────
python3 evaluate_ablation_v2.py "${ARGS[@]}"

echo "Done. Results in $OUTPUT_DIR/"
