#!/bin/bash
# =============================================================================
# run_exp3a_extract.sh — GPU 2: Description extraction (10-12 h, all splits)
#
# Generates thinking-mode descriptions for all 3,628 clips (Train/Valid/Test),
# CLIP-encodes them, and writes to features/text_embeddings_qwen3-vl-30b.h5.
# These feed into Exp 3B (AIM + decoder + MFFM).
#
# Submit with:
#   JOB3A=$(sbatch --parsable slurm/run_exp3a_extract.sh)
#   sbatch --dependency=afterok:${JOB3A} slurm/run_exp3b_aim_mffm.sh
# =============================================================================
#SBATCH --job-name=vars_extract_qwen3_30b
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --nodelist=h86
# NOTE: Qwen3-VL-30B dense weights ~60 GB (bf16) → 2 GPUs; device_map=auto

set -euo pipefail

BASE_DIR="${HOME}/thesis_automatic_faul_recognition/VARS_early_fusion"
# Compact HDF5 (32 frames/clip at 224×224) — no need to decode raw mp4s
HDF5_DIR="${HOME}/data/SoccerNet_HDF5_compact"
HF_CACHE="${HOME}/.cache/huggingface"
FEAT_DIR="${BASE_DIR}/features"

mkdir -p "${BASE_DIR}/slurm_logs" "${FEAT_DIR}"

echo "=== Exp 3A: Extract descriptions (Qwen3-VL-30B, all splits) ==="
echo "Node   : $(hostname)"
echo "GPUs   : ${CUDA_VISIBLE_DEVICES:-<SLURM-assigned>}"
echo "Data   : ${HDF5_DIR}"
echo "Output : ${FEAT_DIR}/text_embeddings_qwen3-vl-30b.h5"
echo "Start  : $(date)"

cd "${BASE_DIR}"

module load uv
uv sync --reinstall
source "${BASE_DIR}/.venv/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Reads 32 pre-extracted frames per clip directly from compact HDF5.
# _iter_actions_hdf5() iterates Train + Valid + Test — no --split needed.
# Resume-safe: already-written action_ids are skipped (HDF5 "a" append mode).
python extract_descriptions.py \
    --vlm_model    qwen3-vl-30b \
    --hdf5_dir     "${HDF5_DIR}" \
    --thinking \
    --output_dir   "${FEAT_DIR}" \
    --hf_cache_dir "${HF_CACHE}"

echo "Done: $(date)"
echo "Output: ${FEAT_DIR}/text_embeddings_qwen3-vl-30b.h5"
