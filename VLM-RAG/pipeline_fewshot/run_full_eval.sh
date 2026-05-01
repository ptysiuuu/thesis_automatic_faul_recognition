#!/bin/bash
#SBATCH --job-name=VLM_Full_Test_Eval
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=VLM_Full_Test_Eval_%j.out

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vlm

cd /net/tscratch/people/plgaszos/sn-mvfoul/VLM-RAG/pipeline_fewshot

TEST_HDF5="/net/tscratch/people/plgaszos/SoccerNet_HDF5/Test.hdf5"
TEST_ANNOTATIONS="/net/tscratch/people/plgaszos/SoccerNet_Data/Test/annotations.json"
LAW12_PDF="/net/tscratch/people/plgaszos/SoccerNet_Data/law12.pdf"
OUTPUT_DIR="test_results_qwen"

echo "Rozpoczynam pełną ewaluację na zbiorze testowym..."

python evaluate_vlm.py \
    --hdf5_path       "$TEST_HDF5" \
    --annotations     "$TEST_ANNOTATIONS" \
    --backend         "qwen" \
    --strategies      few_shot rule_grounded \
    --law12_pdf       "$LAW12_PDF" \
    --frames_per_view 4 \
    --output_dir      "$OUTPUT_DIR"

echo "Ewaluacja zakończona. Wyniki zapisane w katalogu $OUTPUT_DIR"
