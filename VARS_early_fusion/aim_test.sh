#!/bin/bash
#SBATCH --job-name=VARS_aim_decoder
#SBATCH --partition=plgrid-now
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=aim_decoder_%j.out

export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export DASHSCOPE_API_KEY=your_key_here

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python main.py \
    --path             "$DATASET_PATH" \
    --pre_model        aim_vitb16 \
    --pooling_type     transformer \
    --use_decoder \
    --batch_size       2 \
    --accum_steps      4 \
    --LR               5e-5 \
    --weight_decay     1e-3 \
    --max_epochs       2 \
    --patience         10 \
    --num_views        5 \
    --fps              12 \
    --start_frame      58 \
    --end_frame        92 \
    --data_aug         Yes \
    --aug_preset       default \
    --weighted_loss    Yes \
    --balanced_sampler Yes \
    --aux_weight       0.2 \
    --ema_decay        0.999 \
    --freeze_epoch     999 \
    --model_name       VARS_aim_decoder \
    --GPU              0 \
    --max_num_worker   16
