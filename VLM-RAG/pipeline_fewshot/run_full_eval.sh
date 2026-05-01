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

# Ustawienia pamięci dla PyTorcha (przeciwdziałanie fragmentacji dla zmiennych sekwencji)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Konfiguracja cache'a dla HuggingFace
export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

# Konfiguracja środowiska i ścieżek
BACKEND="${BACKEND:-qwen}"
STRATEGIES="${STRATEGIES:-few_shot rule_grounded}" # Możesz dodać inne, ale few_shot jest priorytetem
DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"

# !WAŻNE: Używamy zbioru Test, nie Valid!
HDF5_PATH="/net/tscratch/people/plgaszos/SoccerNet_HDF5/Test.hdf5"
ANNOTATIONS="$DATASET_PATH/Test/annotations.json"
LAW12_PDF="$DATASET_PATH/law12.pdf"
OUTPUT_DIR="test_results_${BACKEND}"

# Aktywacja środowiska Conda
source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm

# Przejście do katalogu z projektem (dodałem wejście do nowego katalogu pipeline_fewshot)
cd /net/tscratch/people/plgaszos/sn-mvfoul/VLM-RAG/pipeline_fewshot

echo "============================================================"
echo "Rozpoczynam pełną ewaluację na zbiorze TESTOWYM (bez limitu próbek)"
echo "Backend: $BACKEND"
echo "Strategie: $STRATEGIES"
echo "Katalog wyjściowy: $OUTPUT_DIR"
echo "============================================================"

# Odpalenie ewaluacji
python evaluate_vlm.py \
    --hdf5_path       "$HDF5_PATH" \
    --annotations     "$ANNOTATIONS" \
    --backend         "$BACKEND" \
    --strategies      $STRATEGIES \
    --law12_pdf       "$LAW12_PDF" \
    --frames_per_view 4 \
    --output_dir      "$OUTPUT_DIR"

echo "Ewaluacja zakończona sukcesem. Wyniki zapisane w $OUTPUT_DIR/"