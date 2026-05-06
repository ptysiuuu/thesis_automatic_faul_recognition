#!/bin/bash
#SBATCH --job-name=VARS_extract_skeletons
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=VARS_extract_skeletons_%x_%j.out

# Ścieżki
DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"
SKELETON_HDF5_ROOT="/net/tscratch/people/plgaszos/SoccerNet_Skeleton_HDF5"

# Konfiguracja środowiska conda
source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate vars

# Zabezpieczenie przed brakiem OpenCV headless
echo "Upewniam się, że wersja numpy i opencv-python-headless są poprawne..."
pip install "numpy==1.26.4" "opencv-python-headless==4.9.0.80" "tqdm" -q

# Przejście do katalogu z kodem
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_stgcn

# Uruchomienie ekstrakcji
echo "Rozpoczynam ekstrakcję szkieletów z $DATASET_PATH do $SKELETON_HDF5_ROOT"

python extract_skeletons.py \
    --data_dir "$DATASET_PATH" \
    --out_dir "$SKELETON_HDF5_ROOT"

echo "Zadanie SLURM zakończone."s
