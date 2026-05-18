#!/bin/bash
#SBATCH --job-name=VARS_explain_video
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=explain_video_%j.out
#
# Smoke-test the video-enabled explainability pipeline on 10 samples.
#
# Required env vars:
#   CHECKPOINT      — path to best_model.pth.tar
#   GOOGLE_API_KEY  — Gemini API key
#
# Optional overrides:
#   SPLIT           — dataset split to run on (default: test)
#   N               — number of samples (default: 10)
#   OUT_DIR         — output directory (default: explain_video_<job_id>)
#   NO_VIDEO        — set to "1" to run image-only baseline instead

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"
CHECKPOINT="${CHECKPOINT:?ERROR: set CHECKPOINT env var to your best model path}"
GOOGLE_API_KEY="${GOOGLE_API_KEY:?ERROR: set GOOGLE_API_KEY env var}"
SPLIT="${SPLIT:-test}"
N="${N:-10}"
OUT_DIR="${OUT_DIR:-explain_video_${SLURM_JOB_ID}}"

VIDEO_FLAG="--use_video"
if [ "${NO_VIDEO:-0}" = "1" ]; then
    VIDEO_FLAG="--no_video"
    OUT_DIR="${OUT_DIR}_novideo"
fi

export GOOGLE_API_KEY

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

echo "=========================================="
echo "VARS Explainability — video smoke test"
echo "  checkpoint : $CHECKPOINT"
echo "  split      : $SPLIT"
echo "  samples    : $N"
echo "  video      : $VIDEO_FLAG"
echo "  output     : $OUT_DIR"
echo "=========================================="

python run_explainability.py \
    --checkpoint     "$CHECKPOINT" \
    --dataset_path   "$DATASET_PATH" \
    --split          "$SPLIT" \
    --fusion_mode \
    --num_views      5 \
    --frames_per_view 16 \
    --fps            25 \
    --start_frame    58 \
    --end_frame      92 \
    --llm_backend    gemini \
    --llm_temperature 0.7 \
    --num_examples   "$N" \
    --max_num_worker 8 \
    --output_dir     "$OUT_DIR" \
    $VIDEO_FLAG

echo "=========================================="
echo "Done. Results in: $OUT_DIR"
echo "  evidence    : $OUT_DIR/evidence_${SPLIT}.json"
echo "  interpreted : $OUT_DIR/interpreted_${SPLIT}.json"
echo "  explanations: $OUT_DIR/explanations_${SPLIT}.json"
echo "=========================================="
