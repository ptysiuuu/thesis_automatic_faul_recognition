#!/bin/bash
#SBATCH --job-name=VARS_temporal_head
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=VARS_temporal_head_%j.out

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"
CHECKPOINT="/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion/models/VARS_tadaformer_b16/5/tadaformer_b16/2e-05/_B2_F1_G0.1_Step3/16_model.pth.tar"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python train_temporal_head.py \
    --mode            full \
    --checkpoint      "$CHECKPOINT" \
    --feature_bank    temporal_tokens.h5 \
    --path            "$DATASET_PATH" \
    --n_passes        10 \
    --num_views       5 \
    --fps             17 \
    --start_frame     63 \
    --end_frame       87 \
    --num_layers      2 \
    --cascade_severity \
    --LR              1e-3 \
    --weight_decay    1e-4 \
    --max_epochs      50 \
    --patience        10 \
    --batch_size      128 \
    --aux_weight      0.2 \
    --ema_decay       0.999 \
    --uncertainty_weighting \
    --model_name      "VARS_temporal_head" \
    --GPU             0 \
    --max_num_worker  16
