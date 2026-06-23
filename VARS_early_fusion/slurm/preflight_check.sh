#!/bin/bash
#SBATCH --job-name=preflight_check
#SBATCH --output=slurm_logs/preflight_%j.out
#SBATCH --error=slurm_logs/preflight_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

BASE_DIR="${HOME}/thesis_automatic_faul_recognition/VARS_early_fusion"
cd "${BASE_DIR}"

module load uv
uv sync --reinstall
source .venv/bin/activate

echo "=========================================="
echo "  CODEBASE INTEGRITY CHECK"
echo "=========================================="

echo ""
echo "--- 1. Required files exist ---"
for f in model.py mvaggregate.py mffm.py task_decoder.py train.py main.py dataset.py data_loader.py tadaformer_backbone.py; do
    if [ -f "$f" ]; then
        echo "  OK   $f"
    else
        echo "  MISSING  $f"
    fi
done

echo ""
echo "--- 2. main.py supports --use_decoder flag ---"
grep -n "use_decoder" main.py || echo "  WARNING: --use_decoder not found in main.py"

echo ""
echo "--- 3. main.py supports --use_mffm flag ---"
grep -n "use_mffm" main.py || echo "  WARNING: --use_mffm not found in main.py"

echo ""
echo "--- 4. TaskQueryDecoder class exists ---"
grep -n "class TaskQueryDecoder" task_decoder.py || echo "  WARNING: TaskQueryDecoder class not found"

echo ""
echo "--- 5. MFFM class exists ---"
grep -n "class MultiModalFeatureFusionModule" mffm.py || echo "  WARNING: MFFM class not found"

echo ""
echo "--- 6. get_tokens method in mvaggregate.py ---"
grep -n "def get_tokens" mvaggregate.py || echo "  WARNING: get_tokens not found in mvaggregate.py"

echo ""
echo "--- 7. MFFM dimension check (768 hardcoded?) ---"
grep -n "768\|feat_dim" mffm.py | head -10

echo ""
echo "--- 8. TaskQueryDecoder accepts variable dim ---"
grep -n "def __init__" task_decoder.py | head -5

echo ""
echo "--- 9. Decoder dim compatibility with TAdaFormer-L/14 (1024) ---"
python3 -c "
from task_decoder import TaskQueryDecoder
d = TaskQueryDecoder(dim=1024, num_layers=2, num_heads=8, dropout=0.1)
print(f'  OK   TaskQueryDecoder(dim=1024) created successfully, params: {sum(p.numel() for p in d.parameters()):,}')
" 2>&1 || echo "  FAILED to create TaskQueryDecoder with dim=1024"

echo ""
echo "--- 10. MFFM with dim=1024 ---"
python3 -c "
from mffm import MultiModalFeatureFusionModule
m = MultiModalFeatureFusionModule(dim=1024, num_layers=2)
print(f'  OK   MFFM(dim=1024) created successfully, params: {sum(p.numel() for p in m.parameters()):,}')
" 2>&1 || echo "  FAILED to create MFFM with dim=1024 — needs code fix"

echo ""
echo "--- 11. train.py handles decoder output format ---"
grep -n "use_decoder\|decoder" train.py | head -10 || echo "  WARNING: no decoder references in train.py"

echo ""
echo "--- 12. Checkpoint files ---"
ls -lh "${BASE_DIR}/checkpoints/" 2>/dev/null || echo "  No checkpoints directory"

echo ""
echo "--- 13. Data files ---"
ls -lh "${HOME}/data/SoccerNet_HDF5_compact_280x490/" 2>/dev/null || echo "  No 280x490 HDF5"
ls -lh "${HOME}/data/SoccerNet_Data/Train/annotations.json" 2>/dev/null || echo "  No Train annotations"
ls -lh "${HOME}/data/SoccerNet_Data/Valid/annotations.json" 2>/dev/null || echo "  No Valid annotations"
ls -lh "${HOME}/data/SoccerNet_Data/Test/annotations.json" 2>/dev/null || echo "  No Test annotations"

echo ""
echo "--- 14. VLM description embeddings ---"
ls -lh "${BASE_DIR}/features/" 2>/dev/null || echo "  No features directory"

echo ""
echo "--- 15. CLIP cache ---"
ls "${HOME}/.cache/huggingface/hub/" 2>/dev/null | grep clip || echo "  No CLIP cache"

echo ""
echo "--- 16. TAdaConv non-square patch ---"
grep -n "spatial_h\|spatial_w" "${HOME}/thesis_automatic_faul_recognition/TAdaConv/tadaconv/models/module_zoo/ops/tadaconv_v2.py" || echo "  WARNING: TAdaConv non-square patch not applied"

echo ""
echo "--- 17. Checkpoint filename hardcoded to 32f ---"
grep -n "k710k400" model.py | head -5

echo ""
echo "=========================================="
echo "  CHECK COMPLETE"
echo "=========================================="
