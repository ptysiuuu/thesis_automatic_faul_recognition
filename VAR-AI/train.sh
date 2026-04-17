#!/bin/bash
#SBATCH --job-name=VAR-AI_VideoMAE
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=VAR-AI_VideoMAE_%j.out
#SBATCH --error=VAR-AI_VideoMAE_%j.err

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
source venv/bin/activate

pip install transformers --quiet

python main.py