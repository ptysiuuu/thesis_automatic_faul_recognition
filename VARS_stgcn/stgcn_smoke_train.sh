#!/bin/bash
#SBATCH --job-name=VARS_stgcn_smoke
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=VARS_stgcn_smoke_%x_%j.out

DATASET_PATH="${DATASET_PATH:-/net/tscratch/people/plgaszos/SoccerNet_Data}"
SKELETON_HDF5_ROOT="${SKELETON_HDF5_ROOT:-/net/tscratch/people/plgaszos/SoccerNet_Skeleton_HDF5}"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_stgcn

python main.py \
    --path                "$DATASET_PATH" \
    --skeleton_hdf5_root  "$SKELETON_HDF5_ROOT" \
    --skeleton_num_joints 18 \
    --skeleton_in_channels 3 \
    --skeleton_layout     coco17 \
    --skeleton_norm       Yes \
    --pre_model           stgcn \
    --pooling_type        transformer \
    --graph_topology      structured \
    --batch_size          32 \
    --accum_steps         1 \
    --LR                  1e-3 \
    --weight_decay        1e-4 \
    --max_epochs          2 \
    --patience            1 \
    --num_views           5 \
    --fps                 17 \
    --start_frame         63 \
    --end_frame           87 \
    --data_aug            No \
    --weighted_loss       Yes \
    --balanced_sampler    Yes \
    --aux_weight          0.2 \
    --ema_decay           0.999 \
    --model_name          "VARS_stgcn_smoke" \
    --GPU                 0 \
    --max_num_worker      8 \
    --only_evaluation     3
