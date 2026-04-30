#!/bin/bash
#SBATCH --job-name=VARS_vlm_eval
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=VARS_vlm_%x_%j.out

# ============================================================
# VLM-based foul classification evaluation
# Tests 4 prompting strategies on the validation set
#
# Usage:
#   BACKEND=qwen MAX_SAMPLES=50 sbatch eval_vlm.sh
#
# For a quick sanity check, set MAX_SAMPLES=20.
# For full val set evaluation, unset MAX_SAMPLES or set to 321.
# ============================================================

BACKEND="${BACKEND:-qwen}"
MAX_SAMPLES="${MAX_SAMPLES:-321}"
STRATEGIES="${STRATEGIES:-zero_shot rule_grounded chain_of_thought few_shot}"
export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME
DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"
HDF5_PATH="/net/tscratch/people/plgaszos/SoccerNet_HDF5/Valid.hdf5"
ANNOTATIONS="$DATASET_PATH/Valid/annotations.json"
LAW12_PDF="$DATASET_PATH/law12.pdf"   # put your FIFA Laws PDF here
OUTPUT_DIR="vlm_results_${BACKEND}"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm

cd /net/tscratch/people/plgaszos/sn-mvfoul/VLM-RAG

# Install dependencies if needed
pip install sentence-transformers pymupdf scikit-learn --quiet

# For Qwen backend:
pip install qwen-vl-utils --quiet

python evaluate_vlm.py \
    --hdf5_path       "$HDF5_PATH" \
    --annotations     "$ANNOTATIONS" \
    --backend         "$BACKEND" \
    --strategies      $STRATEGIES \
    --law12_pdf       "$LAW12_PDF" \
    --frames_per_view 4 \
    --output_dir      "$OUTPUT_DIR" \
    --max_samples     "$MAX_SAMPLES"

echo "Done. Results in $OUTPUT_DIR/"
