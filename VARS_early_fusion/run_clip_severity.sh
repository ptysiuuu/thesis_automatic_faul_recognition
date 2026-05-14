#!/bin/bash
#SBATCH --job-name=VARS_clip_severity
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=clip_severity_%j.out

DATASET_PATH="/net/tscratch/people/plgaszos/SoccerNet_Data"
WORK_DIR="/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion"
EMBEDDINGS_PATH="$WORK_DIR/clip_embedings"

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh

# --- Phase 1: extract CLIP severity embeddings (vlm env) ---
echo "=== Phase 1: Extracting CLIP severity embeddings ==="
conda activate vlm

python "$WORK_DIR/extract_clip_embeddings.py" \
    --output "$EMBEDDINGS_PATH"

if [ ! -f "$EMBEDDINGS_PATH" ]; then
    echo "ERROR: embedding extraction failed, aborting"
    exit 1
fi

echo "=== Phase 1 complete ==="

# --- Phase 2: training with CLIP severity loss (vars env) ---
echo "=== Phase 2: Training with CLIP severity loss ==="
conda activate vars
cd "$WORK_DIR"

python main.py \
    --path                   "$DATASET_PATH" \
    --pre_model              tadaformer_b16 \
    --pooling_type           transformer \
    --batch_size             2 \
    --accum_steps            4 \
    --LR                     5e-5 \
    --weight_decay           1e-3 \
    --max_epochs             20 \
    --patience               7 \
    --num_views              5 \
    --fps                    25 \
    --start_frame            58 \
    --end_frame              92 \
    --data_aug               Yes \
    --aug_preset             default \
    --weighted_loss          Yes \
    --balanced_sampler       Yes \
    --aux_weight             0.2 \
    --ema_decay              0.999 \
    --freeze_epoch           5 \
    --clip_weight            0.1 \
    --clip_embeddings_path   "$EMBEDDINGS_PATH" \
    --model_name             VARS_tadaformer_b16_clip \
    --GPU                    0 \
    --max_num_worker         16

echo "=== Phase 2 complete ==="
