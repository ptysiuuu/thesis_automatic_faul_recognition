#!/bin/bash
#SBATCH --job-name=vars_tada_l14_mvf1_224
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist=h86

set -euo pipefail

BASE_DIR="${HOME}/thesis_automatic_faul_recognition/VARS_early_fusion"
CKPT_DIR="${BASE_DIR}/checkpoints"
DATA_DIR="${HOME}/data/SoccerNet_HDF5_compact"
HF_CACHE="${HOME}/.cache/huggingface"
OUT_DIR="${BASE_DIR}/models"

mkdir -p "${BASE_DIR}/slurm_logs"

echo "=== Exp 1b: TAdaFormer-L/14 16f stride-2 (Phase 1, 224×224) ==="
echo "Node   : $(hostname)"
echo "GPUs   : ${CUDA_VISIBLE_DEVICES:-<SLURM-assigned>}"
echo "Start  : $(date)"

cd "${BASE_DIR}"

module load uv
uv sync --reinstall
source "${BASE_DIR}/.venv/bin/activate"

python main.py \
    --pre_model        tadaformer_l14 \
    --backbone_num_frames 16 \
    --backbone_ckpt_dir "${CKPT_DIR}" \
    --fps              12 \
    --start_frame      58 \
    --end_frame        90 \
    --pooling_type     transformer \
    --batch_size       4 \
    --accum_steps      2 \
    --optimizer        adam \
    --warmup_epochs    5 \
    --LR               1e-5 \
    --weight_decay     0.05 \
    --freeze_epoch     5 \
    --max_epochs       40 \
    --patience         10 \
    --num_views        5 \
    --num_frames       16 \
    --stored_frames    32 \
    --weighted_loss    Yes \
    --balanced_sampler Yes \
    --no_pos_weight \
    --aux_weight       0.2 \
    --ema_decay        0.999 \
    --keep_top_k       3 \
    --model_name       VARS_tada_l14_mvf1_224 \
    --data_dir         "${DATA_DIR}" \
    --output_dir       "${OUT_DIR}" \
    --hf_cache_dir     "${HF_CACHE}" \
    --compact_hdf5 \
    --compact_hdf5_dir "${DATA_DIR}" \
    --max_num_worker   8 \
    --GPU              0

echo "Done: $(date)"
