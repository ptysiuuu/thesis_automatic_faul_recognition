#!/bin/bash
# =============================================================================
# run_ablation.sh
# Run as: sbatch run_ablation.sh <strategy> [max_samples]
#
# Examples:
#   sbatch run_ablation.sh build_medoid_cache     ← run ONCE first
#   sbatch run_ablation.sh static_few_shot
#   sbatch run_ablation.sh data_driven
#   sbatch run_ablation.sh two_stage
#   sbatch run_ablation.sh rag_icl
#   sbatch run_ablation.sh static_few_shot 50     ← quick 50-sample test
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
MAX_SAMPLES="${2:-}"   # empty = full eval

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
OUTPUT_DIR="ablation_results/${STRATEGY}"

# ── Environment ───────────────────────────────────────────────────────────
source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Strategy:    $STRATEGY"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Output:      $OUTPUT_DIR"
echo "============================================================"

# ── Build medoid cache (run once before data_driven / two_stage) ───────────
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
    --hdf5_path        "$EVAL_HDF5"
    --annotations      "$EVAL_ANN"
    --train_annotations "$TRAIN_ANN"
    --law12_pdf        "$LAW12_PDF"
    --faiss_index_path "$FAISS_DIR/train_mvit_features.index"
    --faiss_meta_path  "$FAISS_DIR/train_mvit_metadata.json"
    --medoid_cache     "$MEDOID_CACHE"
    --strategy         "$STRATEGY"
    --frames_per_view  4
    --retrieval_k      3
    --output_dir       "$OUTPUT_DIR"
)

if [ -n "$MAX_SAMPLES" ]; then
    COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python evaluate_ablation.py "${COMMON_ARGS[@]}"

echo "Done. Results in $OUTPUT_DIR/"
