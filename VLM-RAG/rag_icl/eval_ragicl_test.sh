#!/bin/bash
#SBATCH --job-name=vlm_ragicl_test
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=ragicl_test_%j.out

export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HF_HOME

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm
cd /net/tscratch/people/plgaszos/sn-mvfoul/VLM-RAG/rag_icl

FAISS_DIR="/net/tscratch/people/plgaszos/vlm_rag_icl"

python evaluate_ragicl.py \
    --hdf5_path        /net/tscratch/people/plgaszos/SoccerNet_HDF5/Test.hdf5 \
    --annotations      /net/tscratch/people/plgaszos/SoccerNet_Data/Test/annotations.json \
    --law12_pdf        /net/tscratch/people/plgaszos/SoccerNet_Data/law12.pdf \
    --faiss_index_path "$FAISS_DIR/train_mvit_features.index" \
    --faiss_meta_path  "$FAISS_DIR/train_mvit_metadata.json" \
    --strategies       rule_grounded rag_icl \
    --frames_per_view  4 \
    --retrieval_k      3 \
    --output_dir       ragicl_results_test

echo "Test eval done."