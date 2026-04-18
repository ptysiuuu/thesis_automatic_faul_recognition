#!/bin/bash
#SBATCH --job-name=VAR-AI_V2
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=VAR-AI_VideoMAE_%j.out

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet"

python main.py \
    --path          "$DATASET_PATH" \
    --pre_model     mvit_v2_s \
    --pooling_type  transformer \
    --batch_size    4 \
    --LR            1e-4 \
    --weight_decay  1e-3 \
    --max_epochs    40 \
    --patience      8 \
    --num_views     5 \
    --fps           25 \
    --start_frame   0 \
    --end_frame     125 \
    --data_aug      Yes \
    --weighted_loss Yes \
    --balanced_sampler Yes \
    --aux_weight    0.2 \
    --ema_decay     0.999 \
    --use_tta \
    --model_name    VARS_v2 \
    --GPU           0 \
    --max_num_worker 4 \
    --only_evaluation 3
