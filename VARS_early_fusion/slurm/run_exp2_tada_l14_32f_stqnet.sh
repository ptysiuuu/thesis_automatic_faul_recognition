#!/bin/bash
#SBATCH --job-name=tada_l14_32f_stqnet
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist=h86

set -euo pipefail

BASE_DIR="${HOME}/thesis_automatic_faul_recognition/VARS_early_fusion"
CKPT_DIR="${BASE_DIR}/checkpoints"
DATA_DIR="${HOME}/data/SoccerNet_HDF5_compact"
COMPACT_DIR="${DATA_DIR}"
HF_CACHE="${HOME}/.cache/huggingface"
OUT_DIR="${BASE_DIR}/models"

mkdir -p "${BASE_DIR}/slurm_logs"

echo "=== Exp 2: TAdaFormer-L/14 32f — STQNet recipe (AdamW + warmup) ==="
echo "Node   : $(hostname)"
echo "GPUs   : ${CUDA_VISIBLE_DEVICES:-<SLURM-assigned>}"
echo "Start  : $(date)"

cd "${BASE_DIR}"

module load uv
uv sync --reinstall
source "${BASE_DIR}/.venv/bin/activate"

python main.py \
    --pre_model           tadaformer_l14 \
    --backbone_num_frames 32 \
    --backbone_ckpt_dir   "${CKPT_DIR}" \
    --optimizer           adamw \
    --warmup_epochs       5 \
    --LR                  1e-6 \
    --weight_decay        0.05 \
    --freeze_epoch        999 \
    --max_epochs          60 \
    --patience            15 \
    --start_frame         60 \
    --end_frame           92 \
    --fps                 24 \
    --pooling_type        transformer \
    --num_views           5 \
    --batch_size          4 \
    --accum_steps         2 \
    --weighted_loss       Yes \
    --balanced_sampler    Yes \
    --no_pos_weight \
    --aux_weight          0.2 \
    --ema_decay           0.999 \
    --keep_top_k          3 \
    --model_name          VARS_tada_l14_32f_stqnet \
    --path                "${DATA_DIR}" \
    --compact_hdf5_dir    "${COMPACT_DIR}" \
    --compact_hdf5 \
    --output_dir          "${OUT_DIR}" \
    --hf_cache_dir        "${HF_CACHE}" \
    --max_num_worker      8 \
    --GPU                 0

echo "Done: $(date)"
