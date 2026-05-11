#!/bin/bash
# =============================================================================
# run_ablation_v2.sh — SLURM launcher for evaluate_ablation_v2.py
#
# Usage: sbatch run_ablation_v2.sh <strategy> [max_samples] [model_name] [output_subdir] [video_flag]
#
# New severity-focused strategies (all use Qwen2.5-VL-7B):
#   sbatch run_ablation_v2.sh full_sev_two_stage 50      ← smoke test
#   sbatch run_ablation_v2.sh full_sev_two_stage         ← full run
#   sbatch run_ablation_v2.sh per_action_prior   50
#   sbatch run_ablation_v2.sh per_action_prior
#   sbatch run_ablation_v2.sh targeted_retrieval 50
#   sbatch run_ablation_v2.sh targeted_retrieval
#   sbatch run_ablation_v2.sh ordinal_severity   50
#   sbatch run_ablation_v2.sh ordinal_severity
#   sbatch run_ablation_v2.sh cos_full_sev       50      ← best combined
#   sbatch run_ablation_v2.sh cos_full_sev               ← full run
#
# Existing strategies also work (reproduces old results with new code):
#   sbatch run_ablation_v2.sh static_few_shot    50
#   sbatch run_ablation_v2.sh cos_two_stage      50
#
# Video / large-model variants:
#   sbatch run_ablation_v2.sh cos_two_stage 50 Qwen/Qwen2.5-VL-7B-Instruct cos2_video video
#   sbatch run_ablation_v2.sh cos_two_stage 50 Qwen/Qwen2.5-VL-32B-Instruct
# =============================================================================
#SBATCH --job-name=vlm_v2
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=ablation_v2_%x_%j.out

export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HF_HOME

STRATEGY="${1:-static_few_shot}"
MAX_SAMPLES="${2:-}"
MODEL_NAME="${3:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_SUBDIR="${4:-${STRATEGY}}"
VIDEO_FLAG="${5:-}"
VIDEO_MODE=false

if [[ "$VIDEO_FLAG" == "video" || "$VIDEO_FLAG" == "native_video" || "$VIDEO_FLAG" == "true" || "$VIDEO_FLAG" == "1" ]]; then
    VIDEO_MODE=true
fi

DATA_ROOT="/net/tscratch/people/plgaszos/SoccerNet_Data"
HDF5_ROOT="/net/tscratch/people/plgaszos/SoccerNet_HDF5"
FAISS_DIR="/net/tscratch/people/plgaszos/vlm_rag_icl"
SCRIPT_DIR="/net/tscratch/people/plgaszos/sn-mvfoul/VLM-RAG/rag_icl"
OUTPUT_DIR="ablation_results_v2/${OUTPUT_SUBDIR}"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Strategy:    $STRATEGY"
echo "Model:       $MODEL_NAME"
echo "Video mode:  $VIDEO_MODE"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Output:      $OUTPUT_DIR"
echo "Script:      evaluate_ablation_v2.py"
echo "============================================================"

COMMON_ARGS=(
    --hdf5_path         "$HDF5_ROOT/Valid.hdf5"
    --annotations       "$DATA_ROOT/Valid/annotations.json"
    --train_annotations "$DATA_ROOT/Train/annotations.json"
    --law12_pdf         "$DATA_ROOT/law12.pdf"
    --faiss_index_path  "$FAISS_DIR/train_mvit_features.index"
    --faiss_meta_path   "$FAISS_DIR/train_mvit_metadata.json"
    --medoid_cache      "$FAISS_DIR/medoid_cache.json"
    --strategy          "$STRATEGY"
    --model_name        "$MODEL_NAME"
    --frames_per_view   4
    --retrieval_k       3
    --output_dir        "$OUTPUT_DIR"
)

if [ -n "$MAX_SAMPLES" ]; then
    COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

if [[ "$VIDEO_MODE" == true ]]; then
    COMMON_ARGS+=(--video_mode)
fi

python3 evaluate_ablation_v2.py "${COMMON_ARGS[@]}"

echo "Done. Results in $OUTPUT_DIR/"
