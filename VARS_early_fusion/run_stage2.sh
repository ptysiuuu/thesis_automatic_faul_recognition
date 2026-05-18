#!/bin/bash
#SBATCH --job-name=VARS_stage2
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=stage2_%j.out

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"
CHECKPOINT="best_model.pth.tar"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python train_stage2.py \
    --mode            full \
    --checkpoint      "$CHECKPOINT" \
    --feature_bank    stage2_trainvalid.h5 \
    --path            "$DATASET_PATH" \
    --pre_model       tadaformer_b16 \
    --pooling_type    transformer \
    --num_views       5 \
    --fps             16 \
    --start_frame     58 \
    --end_frame       92 \
    --n_passes        10 \
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
    --max_num_worker  16 \
    --model_name      VARS_stage2_trainvalid
