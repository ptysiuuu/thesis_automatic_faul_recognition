#!/bin/bash
#SBATCH --job-name=VARS_smoke_text
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=VARS_smoke_text_%x_%j.out

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"
CHECKPOINT_PATH="/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion/models/VARS_tadaformer_b16/5/tadaformer_b16/2e-05/_B2_F1_G0.1_Step3/5_model.pth.tar"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python main.py \
    --path                "$DATASET_PATH" \
    --pre_model           tadaformer_b16 \
    --pooling_type        transformer \
    --use_text_bridge \
    --path_to_model_weights "$CHECKPOINT_PATH" \
    --only_evaluation     0 \
    --num_views           2 \
    --fps                 10 \
    --start_frame         63 \
    --end_frame           87 \
    --batch_size          1 \
    --max_num_worker      4 \
    --model_name          "VARS_smoke_text" \
    --GPU                 0