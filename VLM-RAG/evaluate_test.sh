#!/bin/bash
# =============================================================================
# Step 3: Full test set evaluation
# sbatch evaluate_test.sh
# =============================================================================
#SBATCH --job-name=vlm_eval_test
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=vlm_eval_%x_%j.out

export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python evaluate_finetuned.py \
    --hdf5_path      /net/tscratch/people/plgaszos/SoccerNet_HDF5/Test.hdf5 \
    --annotations    /net/tscratch/people/plgaszos/SoccerNet_Data/Test/annotations.json \
    --adapter_path   /net/tscratch/people/plgaszos/vlm_finetuned/lora_adapters \
    --base_model     Qwen/Qwen2.5-VL-7B-Instruct \
    --law12_pdf      /net/tscratch/people/plgaszos/SoccerNet_Data/law12.pdf \
    --frames_per_view 4 \
    --output_dir     vlm_test_results

# Run official SoccerNet evaluator on best config
python -c "
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
result = evaluate(
    '/net/tscratch/people/plgaszos/SoccerNet_Data/Test/annotations.json',
    'vlm_test_results/finetuned_rule_grounded_predictions.json'
)
print('Official SoccerNet result:', result)
"
