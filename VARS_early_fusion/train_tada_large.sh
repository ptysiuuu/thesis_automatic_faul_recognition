#!/bin/bash
#SBATCH --job-name=VARS_tada_l14
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=tada_l14_%j.out

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"
export TORCH_HOME=/net/people/plgrid/plgaszos/.cache/torch

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars

cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python main.py \
    --path             "$DATASET_PATH" \
    --pre_model        tadaformer_l14 \
    --pooling_type     transformer \
    --batch_size       2 \
    --accum_steps      4 \
    --LR               3e-5 \
    --weight_decay     1e-3 \
    --max_epochs       30 \
    --patience         8 \
    --num_views        4 \
    --fps              12 \
    --start_frame      58 \
    --end_frame        92 \
    --data_aug         Yes \
    --aug_preset       default \
    --weighted_loss    Yes \
    --balanced_sampler Yes \
    --no_pos_weight \
    --aux_weight       0.2 \
    --ema_decay        0.999 \
    --freeze_epoch     5 \
    --model_name       VARS_tada_l14 \
    --GPU              0 \
    --max_num_worker   16