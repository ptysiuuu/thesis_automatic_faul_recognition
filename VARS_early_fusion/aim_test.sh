#!/bin/bash
#SBATCH --job-name=aim_verify
#SBATCH --partition=plgrid-now
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=aim_verify_%j.out

source /net/people/plgrid/plgaszos/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgaszos/conda_envs/vlm32b
cd /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion

python - <<'EOF'
import torch
from aim_backbone import AIMBackbone

CKPT = "/net/tscratch/people/plgaszos/sn-mvfoul/checkpoints/vit_b_clip_16frame_k400.pth"

print("=== Loading AIMBackbone ===")
m = AIMBackbone(CKPT).cuda()

trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in m.parameters() if not p.requires_grad)
print(f"Trainable : {trainable/1e6:.1f}M")
print(f"Frozen    : {frozen/1e6:.1f}M")
print(f"Expected  : ~11M trainable, ~85M frozen")

# Check a known ViT weight is actually loaded (not random)
vit_weight = m._vit.class_embedding
print(f"\nclass_embedding norm: {vit_weight.norm().item():.4f}")
print(f"Expected: non-trivial value (>>0). If ~0 or random: loading failed.")

# Forward pass
print("\n=== Forward pass ===")
dummy = torch.randn(2, 3, 16, 224, 224).cuda()
with torch.no_grad():
    out = m(dummy)
print(f"Output shape: {out.shape}  (expected [2, 16, 768])")

# Check trainable param names (should be adapters only)
print("\n=== Trainable param names (first 5) ===")
trainable_names = [n for n, p in m.named_parameters() if p.requires_grad]
for n in trainable_names[:5]:
    print(" ", n)
print(f"  ... ({len(trainable_names)} total trainable params)")

print("\n=== DONE ===")
EOF