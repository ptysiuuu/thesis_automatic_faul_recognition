#!/bin/bash
#SBATCH --job-name=Tada_deter
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=tada_deter_%j.out

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python main.py \
    --path             "$DATASET_PATH" \
    --pre_model        tadaformer_b16 \
    --pooling_type     transformer \
    --batch_size       2 \
    --accum_steps      4 \
    --LR               5e-5 \
    --weight_decay     1e-3 \
    --max_epochs       20 \
    --patience         7 \
    --num_views        5 \
    --fps              12 \
    --start_frame      58 \
    --end_frame        92 \
    --data_aug         Yes \
    --weighted_loss    Yes \
    --balanced_sampler Yes \
    --aux_weight       0.2 \
    --ema_decay        0.999 \
    --freeze_epoch     5 \
    --model_name       VARS_tadaformer_b16_deter \
    --GPU              0 \
    --max_num_worker   16