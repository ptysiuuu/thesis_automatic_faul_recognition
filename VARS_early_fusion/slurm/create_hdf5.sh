#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodelist=h86
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/create_hdf5_%j.out

module load uv
cd ~/thesis_automatic_faul_recognition/VARS_early_fusion
uv sync --reinstall
source .venv/bin/activate

python create_compact_hdf5.py \
    --data_dir    ~/data/SoccerNet_Data \
    --output_dir  ~/data/SoccerNet_HDF5_compact \
    --num_frames  32 \
    --start_frame 58 \
    --end_frame   92 \
    --img_size    224 \
    --splits      Train Valid Test \
    --num_workers 16
