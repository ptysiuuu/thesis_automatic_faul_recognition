#!/bin/bash
# =============================================================================
# run_ablation.sh
# Run as: sbatch run_ablation.sh <strategy> [max_samples] [model_name] [output_dir]
#
# Arguments:
#   $1  strategy      — one of: build_medoid_cache, static_few_shot, data_driven,
#                       two_stage, rag_icl, cos_two_stage
#   $2  max_samples   — integer or empty string "" for full eval
#   $3  model_name    — HuggingFace model ID (default: Qwen/Qwen2.5-VL-7B-Instruct)
#   $4  output_dir    — override output directory (default: ablation_results/<strategy>)
#
# Examples:
#   sbatch run_ablation.sh build_medoid_cache
#
#   # Standard ablation rows (Qwen2.5-VL-7B baseline)
#   sbatch run_ablation.sh static_few_shot
#   sbatch run_ablation.sh data_driven
#   sbatch run_ablation.sh two_stage
#   sbatch run_ablation.sh rag_icl
#   sbatch run_ablation.sh cos_two_stage
#
#   # Quick 50-sample smoke tests
#   sbatch run_ablation.sh static_few_shot 50
#   sbatch run_ablation.sh cos_two_stage   50
#
#   # Model comparison: Video-R1 with same strategies
#   sbatch run_ablation.sh static_few_shot "" Video-R1/Video-R1-7B ablation_results/video_r1_static_few_shot
#   sbatch run_ablation.sh cos_two_stage   "" Video-R1/Video-R1-7B ablation_results/video_r1_cos_two_stage
#
#   # Model comparison smoke tests
#   sbatch run_ablation.sh static_few_shot 50 Video-R1/Video-R1-7B ablation_results/video_r1_static_few_shot_50
#   sbatch run_ablation.sh cos_two_stage   50 Video-R1/Video-R1-7B ablation_results/video_r1_cos_two_stage_50
# =============================================================================
#SBATCH --job-name=vlm_ablation
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=ablation_%x_%j.out

# ── Paths ─────────────────────────────────────────────────────────────────
export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HF_HOME

STRATEGY="${1:-static_few_shot}"
MAX_SAMPLES="${2:-}"
MODEL_NAME="${3:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_DIR="${4:-ablation_results/${STRATEGY}}"

DATA_ROOT="/net/tscratch/people/plgaszos/SoccerNet_Data"
HDF5_ROOT="/net/tscratch/people/plgaszos/SoccerNet_HDF5"
FAISS_DIR="/net/tscratch/people/plgaszos/vlm_rag_icl"
SCRIPT_DIR="/net/tscratch/people/plgaszos/sn-mvfoul/VLM-RAG/rag_icl"

EVAL_HDF5="$HDF5_ROOT/Valid.hdf5"
EVAL_ANN="$DATA_ROOT/Valid/annotations.json"
TRAIN_HDF5="$HDF5_ROOT/Train.hdf5"
TRAIN_ANN="$DATA_ROOT/Train/annotations.json"
LAW12_PDF="$DATA_ROOT/law12.pdf"
MEDOID_CACHE="$FAISS_DIR/medoid_cache.json"

# ── Environment ───────────────────────────────────────────────────────────
source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Strategy:    $STRATEGY"
echo "Model:       $MODEL_NAME"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Output:      $OUTPUT_DIR"
echo "============================================================"

# ── Build medoid cache (run once before data_driven / two_stage / cos_two_stage)
if [ "$STRATEGY" = "build_medoid_cache" ]; then
    python evaluate_ablation.py \
        --build_medoid_cache \
        --train_hdf5        "$TRAIN_HDF5" \
        --train_annotations "$TRAIN_ANN" \
        --faiss_index_path  "$FAISS_DIR/train_mvit_features.index" \
        --faiss_meta_path   "$FAISS_DIR/train_mvit_metadata.json" \
        --medoid_cache      "$MEDOID_CACHE"
    echo "Medoid cache built at $MEDOID_CACHE"
    exit 0
fi

# ── Build common args ─────────────────────────────────────────────────────
COMMON_ARGS=(
    --hdf5_path         "$EVAL_HDF5"
    --annotations       "$EVAL_ANN"
    --train_annotations "$TRAIN_ANN"
    --law12_pdf         "$LAW12_PDF"
    --faiss_index_path  "$FAISS_DIR/train_mvit_features.index"
    --faiss_meta_path   "$FAISS_DIR/train_mvit_metadata.json"
    --medoid_cache      "$MEDOID_CACHE"
    --strategy          "$STRATEGY"
    --model_name        "$MODEL_NAME"
    --frames_per_view   4
    --retrieval_k       3
    --output_dir        "$OUTPUT_DIR"
)

if [ -n "$MAX_SAMPLES" ]; then
    COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python evaluate_ablation.py "${COMMON_ARGS[@]}"

echo "Done. Results in $OUTPUT_DIR/"