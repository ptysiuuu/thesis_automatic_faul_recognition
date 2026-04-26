#!/bin/bash
#SBATCH --job-name=VARS_graph2
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=VARS_graph2_%x_%j.out

# Full ablation experiment:
#   GAT fully_connected aggregator
#   Cascaded severity head (action logits → concat → severity head)
#   Uncertainty weighting (Kendall & Gal, 2018) on severity + action losses
#   Backbone frozen for 10 epochs, then fine-tuned at 1e-5
#   Patience 12

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python main.py \
    --path                "$DATASET_PATH" \
    --pre_model           mvit_v2_s \
    --pooling_type        gat \
    --graph_topology      fully_connected \
    --cascade_severity \
    --uncertainty_weighting \
    --freeze_epoch        10 \
    --batch_size          4 \
    --accum_steps         1 \
    --LR                  1e-4 \
    --weight_decay        1e-3 \
    --max_epochs          50 \
    --patience            12 \
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
    --model_name          "VARS_graph2_cascade_uw" \
    --GPU                 0 \
    --max_num_worker      16 \
    --only_evaluation     3
