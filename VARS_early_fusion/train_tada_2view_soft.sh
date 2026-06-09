#!/bin/bash
#SBATCH --job-name=VARS_tada_2view_soft
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=tada_2view_soft_%j.out

# Activate conda environment vars
module load miniconda3
source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars

# Set working directory
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

# Set HF cache directories
export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=/net/tscratch/people/plgaszos/.cache/huggingface

python3 main.py \
  --path /net/tscratch/people/plgaszos/SoccerNet_Data \
  --pre_model tadaformer_b16 \
  --pooling_type transformer \
  --batch_size 2 \
  --accum_steps 4 \
  --LR 5e-5 \
  --weight_decay 1e-3 \
  --max_epochs 30 \
  --patience 10 \
  --num_views 5 \
  --fps 12 \
  --start_frame 58 \
  --end_frame 92 \
  --data_aug Yes \
  --aug_preset default \
  --weighted_loss Yes \
  --balanced_sampler Yes \
  --aux_weight 0.2 \
  --ema_decay 0.999 \
  --freeze_epoch 999 \
  --model_name VARS_tada_2view_soft \
  --GPU 0 \
  --max_num_worker 16
