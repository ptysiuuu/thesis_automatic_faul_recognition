#!/bin/bash
#SBATCH --job-name=desc_valid_test
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=desc_valid_test_%j.out

export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vlm32b
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python extract_descriptions.py \
    --data_root   /net/tscratch/people/plgaszos/SoccerNet_Data \
    --output_hdf5 features/text_embeddings.h5 \
    --output_json features/text_descriptions_valid_test.json \
    --quantization none \
    --num_frames  8 \
    --max_new_tokens 50 \
    --timeout_s   60