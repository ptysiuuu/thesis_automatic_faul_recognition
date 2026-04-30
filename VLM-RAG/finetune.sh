#!/bin/bash
# =============================================================================
# Step 2: LoRA finetuning (GPU job)
# sbatch finetune.sh
# =============================================================================
#SBATCH --job-name=vlm_finetune
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=vlm_finetune_%x_%j.out

export HF_HOME=/net/tscratch/people/plgaszos/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm
cd /net/tscratch/people/plgaszos/sn-mvfoul/VLM-RAG

pip install peft trl qwen-vl-utils --quiet

python finetune_vlm.py \
    --dataset_dir  /net/tscratch/people/plgaszos/vlm_dataset \
    --output_dir   /net/tscratch/people/plgaszos/vlm_finetuned \
    --model_name   Qwen/Qwen2.5-VL-7B-Instruct \
    --lora_rank    16 \
    --max_epochs   5 \
    --batch_size   1 \
    --grad_accum   8 \
    --lr           2e-4

echo "Finetuning done."

---