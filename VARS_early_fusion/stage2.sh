#!/bin/bash
#SBATCH --job-name=VARS_stage2
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=VARS_stage2_%x_%j.out

# Stage-2 feature-bank training.
#
# Usage:
#   CHECKPOINT=models/VARS_gat_structured/... sbatch stage2.sh
#
# Override defaults with env vars:
#   CHECKPOINT   — path to best_model.pth.tar  (required)
#   TOPOLOGY     — graph topology (default: structured)
#   POOLING      — aggregation type (default: gat)
#   N_PASSES     — augmented passes for extraction (default: 10)
#   MODEL_NAME   — output directory (default: VARS_stage2)
#   CASCADE      — set to "--cascade_severity" to enable (default: empty)
#   UW           — set to "--uncertainty_weighting" to enable (default: empty)

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"
CHECKPOINT="${CHECKPOINT:?ERROR: set CHECKPOINT env var to your best model path}"
TOPOLOGY="${TOPOLOGY:-structured}"
POOLING="${POOLING:-gat}"
N_PASSES="${N_PASSES:-10}"
MODEL_NAME="${MODEL_NAME:-VARS_stage2}"
CASCADE="${CASCADE:-}"
UW="${UW:-}"
FEATURE_BANK="${MODEL_NAME}/features.h5"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python train_stage2.py \
    --mode            full \
    --checkpoint      "$CHECKPOINT" \
    --feature_bank    "$FEATURE_BANK" \
    --path            "$DATASET_PATH" \
    --n_passes        "$N_PASSES" \
    --pre_model       mvit_v2_s \
    --pooling_type    "$POOLING" \
    --graph_topology  "$TOPOLOGY" \
    --num_views       5 \
    --fps             17 \
    --start_frame     63 \
    --end_frame       87 \
    --batch_size      8 \
    --head_batch_size 512 \
    --LR              1e-3 \
    --weight_decay    1e-4 \
    --max_epochs      30 \
    --patience        6 \
    --aux_weight      0.2 \
    --ema_decay       0.999 \
    --accum_steps     1 \
    --GPU             0 \
    --max_num_worker  4 \
    --model_name      "$MODEL_NAME" \
    $CASCADE \
    $UW
