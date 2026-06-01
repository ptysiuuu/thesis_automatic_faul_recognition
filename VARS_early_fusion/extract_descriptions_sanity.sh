#!/bin/bash
#SBATCH --job-name=VARS_desc_sanity
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=extract_descriptions_sanity_%j.out

export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

HDF5_ROOT="/net/tscratch/people/plgaszos/SoccerNet_HDF5"
OUT_DIR="/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion/features"
OUTPUT_H5="${OUT_DIR}/text_embeddings_sanity.h5"
OUTPUT_JSON="${OUT_DIR}/text_descriptions_sanity.json"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm32b
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python extract_descriptions.py \
    --data_root /net/tscratch/people/plgaszos/SoccerNet_Data \
    --output_hdf5 features/text_embeddings_sanity.h5 \
    --output_json features/text_descriptions_sanity.json \
    --quantization none \
	--max_new_tokens 60 \
    --num_frames 8 \
    --max_actions 5 \
    --timeout_s 60
