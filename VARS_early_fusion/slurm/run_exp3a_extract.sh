#!/bin/bash
#SBATCH --job-name=vars_extract_qwen3_30b
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --nodelist=h86
# NOTE: Qwen3-VL-30B needs ~60 GB VRAM; 2 GPUs requested for headroom

set -euo pipefail

BASE_DIR="${HOME}/thesis_automatic_faul_recognition/VARS_early_fusion"
DATA_DIR="${HOME}/data/SoccerNet_HDF5"
HF_CACHE="${HOME}/.cache/huggingface"
FEAT_DIR="${BASE_DIR}/features"

mkdir -p "${BASE_DIR}/slurm_logs" "${FEAT_DIR}"

echo "=== Exp 3A: Extract descriptions (Qwen3-VL-30B) ==="
echo "Node   : $(hostname)"
echo "GPUs   : ${CUDA_VISIBLE_DEVICES:-<SLURM-assigned>}"
echo "Start  : $(date)"

cd "${BASE_DIR}"

module load uv
uv sync --reinstall
source "${BASE_DIR}/.venv/bin/activate"

# Extract for Train, Valid, and Test splits
for SPLIT in Train Valid Test; do
    echo "--- Extracting split: ${SPLIT} ---"
    python extract_descriptions.py \
        --vlm_model    qwen3-vl-30b \
        --data_dir     "${DATA_DIR}" \
        --split        "${SPLIT}" \
        --thinking \
        --output_dir   "${FEAT_DIR}" \
        --hf_cache_dir "${HF_CACHE}"
done

echo "Done: $(date)"
echo "Output: ${FEAT_DIR}/text_embeddings_qwen3-vl-30b.h5"
