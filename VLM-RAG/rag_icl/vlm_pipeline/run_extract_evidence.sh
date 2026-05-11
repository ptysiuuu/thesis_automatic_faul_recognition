#!/bin/bash
# =============================================================================
# run_extract_evidence.sh — SLURM launcher for extract_evidence.py
#
# Usage: sbatch run_extract_evidence.sh [max_samples] [model_name] [output_subdir] [video_flag]
#
# Examples:
#   sbatch run_extract_evidence.sh 50 Qwen/Qwen2.5-VL-7B-Instruct smoke_out video
#   sbatch run_extract_evidence.sh  Qwen/Qwen2.5-VL-7B-Instruct full_out
# =============================================================================
#SBATCH --job-name=vlm_extract
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=extract_evidence_%x_%j.out

export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HF_HOME

MAX_SAMPLES="${1:-}"
SPLIT="${2:-valid}"
MODEL_NAME="${3:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_SUBDIR="${4:-evidence}"
VIDEO_FLAG="${5:-}"
VIDEO_MODE=false

if [[ "$VIDEO_FLAG" == "video" || "$VIDEO_FLAG" == "native_video" || "$VIDEO_FLAG" == "true" || "$VIDEO_FLAG" == "1" ]]; then
    VIDEO_MODE=true
fi

DATA_ROOT="/net/tscratch/people/plgaszos/SoccerNet_Data"
HDF5_ROOT="/net/tscratch/people/plgaszos/SoccerNet_HDF5"
SCRIPT_DIR="/net/tscratch/people/plgaszos/sn-mvfoul/VLM-RAG/rag_icl"
OUTPUT_DIR="evidence_extraction/${OUTPUT_SUBDIR}"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm32b
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Split:       $SPLIT"
echo "Model:       $MODEL_NAME"
echo "Video mode:  $VIDEO_MODE"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Output:      $OUTPUT_DIR"
echo "Script:      extract_evidence.py"
echo "============================================================"

# Compute split-specific paths
split_lc=$(echo "$SPLIT" | tr '[:upper:]' '[:lower:]')
if [[ "$split_lc" == "train" ]]; then
    split_cap="Train"
elif [[ "$split_lc" == "valid" || "$split_lc" == "val" ]]; then
    split_cap="Valid"
else
    split_cap="Test"
fi

hdf5_path="$HDF5_ROOT/${split_cap}.hdf5"
annotations_path="$DATA_ROOT/${split_cap}/annotations.json"

COMMON_ARGS=(
    --hdf5_path         "$hdf5_path"
    --annotations       "$annotations_path"
    --split             "$split_lc"
    --model_name        "$MODEL_NAME"
    --output_dir        "$OUTPUT_DIR"
    --frames_per_view   4
)

if [ -n "$MAX_SAMPLES" ]; then
    COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

if [[ "$VIDEO_MODE" == true ]]; then
    echo "Enabling video mode via backend flag"
    # The extraction script forces use_video_mode internally; keep for clarity
fi

python3 -m vlm_pipeline.extract_evidence "${COMMON_ARGS[@]}"

echo "Done. Results in $OUTPUT_DIR/"
