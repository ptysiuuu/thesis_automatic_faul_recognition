#!/bin/bash
#SBATCH --job-name=extract_desc_qwen3
#SBATCH --output=slurm_logs/extract_desc_%j.out
#SBATCH --error=slurm_logs/extract_desc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist=h86

BASE_DIR="${HOME}/thesis_automatic_faul_recognition/VARS_early_fusion"

module load uv
cd "${BASE_DIR}"
uv sync --reinstall
source .venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python extract_descriptions.py \
    --hdf5_dir      "${HOME}/data/SoccerNet_HDF5_compact" \
    --output_dir    "${BASE_DIR}/features" \
    --vlm_model     qwen3-vl-30b-a3b \
    --thinking \
    --num_frames    8 \
    --max_new_tokens 256 \
    --timeout_s     90