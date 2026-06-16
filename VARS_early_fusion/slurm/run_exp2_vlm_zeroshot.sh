#!/bin/bash
#SBATCH --job-name=vars_vlm_zeroshot
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --nodelist=h86
# NOTE: --gres=gpu:2 because Qwen3-VL-30B needs ~60 GB VRAM (2x 96 GB to be safe)

set -euo pipefail

BASE_DIR="${HOME}/thesis_automatic_faul_recognition/VARS_early_fusion"
DATA_DIR="${HOME}/data/SoccerNet_HDF5"
HF_CACHE="${HOME}/.cache/huggingface"
RESULTS_DIR="${BASE_DIR}/results"

mkdir -p "${BASE_DIR}/slurm_logs" "${RESULTS_DIR}"

echo "=== Exp 2: Qwen3-VL-30B Zero-Shot ==="
echo "Node   : $(hostname)"
echo "GPUs   : ${CUDA_VISIBLE_DEVICES:-<SLURM-assigned>}"
echo "Start  : $(date)"

cd "${BASE_DIR}"

module load uv
uv sync --reinstall
source "${BASE_DIR}/.venv/bin/activate"

python vlm_zeroshot.py \
    --vlm_model    qwen3-vl-30b \
    --data_dir     "${DATA_DIR}" \
    --split        Test \
    --thinking \
    --num_frames   8 \
    --output_path  "${RESULTS_DIR}/vlm_zeroshot_qwen3_30b_test.json" \
    --hf_cache_dir "${HF_CACHE}"

echo "Done: $(date)"
