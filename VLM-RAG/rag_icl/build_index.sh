#!/bin/bash
#SBATCH --job-name=vlm_build_index
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=build_index_%j.out

export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm
cd /net/tscratch/people/plgaszos/sn-mvfoul/VLM-RAG/rag_icl

pip install faiss-gpu --quiet

python build_faiss_index.py \
    --hdf5_path   /net/tscratch/people/plgaszos/SoccerNet_HDF5/Train.hdf5 \
    --annotations /net/tscratch/people/plgaszos/SoccerNet_Data/Train/annotations.json \
    --output_dir  /net/tscratch/people/plgaszos/vlm_rag_icl

echo "FAISS index built."