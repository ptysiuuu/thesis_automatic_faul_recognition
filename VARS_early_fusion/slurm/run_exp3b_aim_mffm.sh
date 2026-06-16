#!/bin/bash
#SBATCH --job-name=vars_aim_decoder_mffm
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist=h86
# NOTE: submit with dependency on Exp 3A:
#   JOB3A=$(sbatch --parsable run_exp3a_extract.sh)
#   sbatch --dependency=afterok:${JOB3A} run_exp3b_aim_mffm.sh

set -euo pipefail

BASE_DIR="${HOME}/thesis_automatic_faul_recognition/VARS_early_fusion"
DATA_DIR="${HOME}/data/SoccerNet_HDF5"
HF_CACHE="${HOME}/.cache/huggingface"
FEAT_DIR="${BASE_DIR}/features"
OUT_DIR="${BASE_DIR}/models"

DESCRIPTIONS="${FEAT_DIR}/text_embeddings_qwen3-vl-30b.h5"

mkdir -p "${BASE_DIR}/slurm_logs"

echo "=== Exp 3B: AIM + Decoder + MFFM (Qwen3-VL-30B descriptions) ==="
echo "Node            : $(hostname)"
echo "GPUs            : ${CUDA_VISIBLE_DEVICES:-<SLURM-assigned>}"
echo "Descriptions    : ${DESCRIPTIONS}"
echo "Start           : $(date)"

if [[ ! -f "${DESCRIPTIONS}" ]]; then
    echo "ERROR: descriptions file not found: ${DESCRIPTIONS}"
    echo "Run Exp 3A first (or submit with --dependency=afterok:<3A_job_id>)"
    exit 1
fi

cd "${BASE_DIR}"

module load uv
uv sync --reinstall
source "${BASE_DIR}/.venv/bin/activate"

python main.py \
    --pre_model         aim_vitb16 \
    --pooling_type      transformer \
    --use_decoder \
    --use_mffm \
    --descriptions_path "${DESCRIPTIONS}" \
    --batch_size        8 \
    --accum_steps       1 \
    --LR                1e-5 \
    --weight_decay      1e-3 \
    --freeze_epoch      999 \
    --max_epochs        60 \
    --patience          12 \
    --num_views         5 \
    --weighted_loss     Yes \
    --balanced_sampler  Yes \
    --no_pos_weight \
    --aux_weight        0.2 \
    --ema_decay         0.999 \
    --keep_top_k        3 \
    --model_name        VARS_aim_decoder_mffm_qwen3 \
    --data_dir          "${DATA_DIR}" \
    --output_dir        "${OUT_DIR}" \
    --hf_cache_dir      "${HF_CACHE}" \
    --GPU               0

echo "Done: $(date)"
