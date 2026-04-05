#!/bin/bash
#SBATCH --job-name=VARS_Transformer
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=transformer_results_part2.log

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
  --weighted_loss "Yes" \
  --path_to_model_weights "models/VARS/5/mvit_v2_s/0.0001/_B4_F16_S_G0.5_Step20/14_model.pth.tar"