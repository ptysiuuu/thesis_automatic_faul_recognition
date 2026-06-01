#!/bin/bash
#SBATCH --job-name=VARS_tada_decoder_mffm
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=tada_decoder_mffm_%j.out

export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"
DESC_PATH="/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion/features/text_embeddings.h5"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python main.py \
    --path              "$DATASET_PATH" \
    --pre_model         tadaformer_b16 \
    --pooling_type      transformer \
    --use_decoder \
    --use_mffm \
    --descriptions_path "$DESC_PATH" \
    --batch_size        2 \
    --accum_steps       4 \
    --LR                5e-5 \
    --weight_decay      1e-3 \
    --max_epochs        20 \
    --patience          7 \
    --num_views         5 \
    --fps               12 \
    --start_frame       58 \
    --end_frame         92 \
    --data_aug          Yes \
    --aug_preset        default \
    --weighted_loss     Yes \
    --balanced_sampler  Yes \
    --aux_weight        0.2 \
    --ema_decay         0.999 \
    --freeze_epoch      5 \
    --model_name        VARS_tadaformer_decoder_mffm \
    --GPU               0 \
    --max_num_worker    16