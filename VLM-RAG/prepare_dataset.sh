#!/bin/bash
# =============================================================================
# Step 1: Prepare dataset (CPU job, no GPU needed)
# sbatch prepare_dataset.sh
# =============================================================================
#SBATCH --job-name=vlm_prep
#SBATCH --partition=plgrid-cpu
#SBATCH --account=plggolemml26-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=vlm_prep_%x_%j.out
export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

pip install sentence-transformers pymupdf --quiet

python prepare_vlm_dataset.py \
    --hdf5_root       /net/tscratch/people/plgaszos/SoccerNet_HDF5 \
    --data_root       /net/tscratch/people/plgaszos/SoccerNet_Data \
    --output_dir      /net/tscratch/people/plgaszos/vlm_dataset \
    --frames_per_view 4 \
    --strategy        rule_grounded \
    --law12_pdf       /net/tscratch/people/plgaszos/SoccerNet_Data/law12.pdf

echo "Dataset prepared."

---