#!/bin/bash
#SBATCH --job-name=VARS_tadaformer_text
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=22:00:00
#SBATCH --output=VARS_tadaformer_text_%x_%j.out

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"
CHECKPOINT_PATH="/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion/models/VARS_tadaformer_b16/5/tadaformer_b16/2e-05/_B2_F1_G0.1_Step3/best_model.pth.tar"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python main.py \
    --path                "$DATASET_PATH" \
    --pre_model           tadaformer_b16 \
    --pooling_type        transformer \
	--cascade_severity \
    --use_text_bridge \
    --continue_training \
    --path_to_model_weights "$CHECKPOINT_PATH" \
    --freeze_epoch        0 \
    --uncertainty_weighting \
    --batch_size          4 \
    --accum_steps         1 \
    --LR                  1e-4 \
    --weight_decay        5e-3 \
    --max_epochs          60 \
    --patience            15 \
    --num_views           5 \
    --fps                 17 \
    --start_frame         63 \
    --end_frame           87 \
    --data_aug            Yes \
    --weighted_loss       Yes \
    --balanced_sampler    Yes \
    --aux_weight          0.2 \
    --ema_decay           0.999 \
    --use_tta \
    --model_name          "VARS_tadaformer_text" \
    --GPU                 0 \
    --max_num_worker      16 \
    --only_evaluation     3
