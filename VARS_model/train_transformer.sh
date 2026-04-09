#!/bin/bash
#SBATCH --job-name=VARS_step2_focal
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=step6_modaldrop_classifydrop.log

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_model

python main.py \
  --pooling_type "transformer" \
  --pre_model "mvit_v2_s" \
  --path "/net/tscratch/people/plgaszos/SoccerNet_Data" \
  --start_frame 63 \
  --end_frame 87 \
  --fps 17 \
  --batch_size 4 \
  --only_evaluation 3 \
  --max_num_worker 15 \
  --step_size 20 \
  --gamma 0.5 \
  --max_epochs 60 \
  --weight_decay 0.001 \
  --model_name "VARS_step6_dropout_both"
