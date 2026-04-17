#!/bin/bash
#SBATCH --job-name=VAR-AI_train
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=VAR-AI_1

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars
source venv/bin/activate

python main.py --net_name r2plus1d_18 --rule_weight 0.05