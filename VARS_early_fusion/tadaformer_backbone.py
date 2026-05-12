"""
tadaformer_backbone.py
======================
TAdaFormer-B/16 as a drop-in backbone for MVAggregate.

Input  : [B, C, T, H, W]
Output : [B, 768]

Notes:
- Optionally renormalizes from MViT/ImageNet stats to CLIP stats
- num_frames=16, tublet_stride=2 → 8 temporal tokens
- forward() returns [B, 768] via CLS token mean across temporal dim
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

TADACONV_ROOT = "/net/tscratch/people/plgaszos/sn-mvfoul/TAdaConv"
if TADACONV_ROOT not in sys.path:
    sys.path.insert(0, TADACONV_ROOT)


# ---------------------------------------------------------------------------
# Minimal config object that mimics the CfgNode interface TAdaConv expects
# ---------------------------------------------------------------------------


class _Cfg:
    """Nested attribute config — mirrors fvcore/yacs CfgNode access pattern."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, _Cfg(v) if isinstance(v, dict) else v)


def _make_b16_cfg(num_frames=16, num_classes=0, drop_path=0.1):
    return _Cfg(
        {
            "VIDEO": {
                "BACKBONE": {
                    "META_ARCH": "VisionTransformer",
                    "INPUT_RES": 224,
                    "PATCH_SIZE": 16,
                    "TUBLET_SIZE": 3,
                    "TUBLET_STRIDE": 2,
                    "NUM_FEATURES": 768,
                    "NUM_OUT_FEATURES": 768,
                    "DEPTH": 12,
                    "NUM_HEADS": 12,
                    "DROP_PATH": drop_path,
                    "ATTN_DROPOUT": 0.0,
                    "REQUIRE_PROJ": False,
                    "ATTN_MASK_ENABLE": False,
                    "DOUBLE_TADA": False,
                    "FREEZE": False,
                    "REDUCTION": 2,
                    "TEMP_ENHANCE": False,
                    "BRANCH": {
                        "NAME": "TAdaFormerBlock",
                        "ROUTE_FUNC_K": [3, 3],
                        "ROUTE_FUNC_R": 2,
                    },
                },
                "HEAD": {
                    "NAME": "BaseHead",
                    "OUTPUT_DIM": 512,
                    "NUM_CLASSES": num_classes,
                    "DROPOUT_RATE": 0.5,
                },
            },
            "DATA": {
                "NUM_INPUT_FRAMES": num_frames,
            },
        }
    )


# ---------------------------------------------------------------------------
# Backbone wrapper
# ---------------------------------------------------------------------------


class TAdaFormerBackbone(nn.Module):
    """
    Wraps TAdaFormer-B/16 for use as a drop-in backbone in MVAggregate.

    Args:
        checkpoint_path : path to downloaded .pyth checkpoint
        num_frames      : must match the checkpoint (16 for K400/K710 ckp)
        drop_path       : stochastic depth rate (0.1 for fine-tuning)
    """

    # CLIP normalisation constants (from tadaformer_b16_k400_16f.yaml)
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]

    def __init__(
        self,
        checkpoint_path: str,
        num_frames: int = 16,
        drop_path: float = 0.1,
        apply_renormalize: bool = True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.feat_dim = 768
        self.apply_renormalize = apply_renormalize
        self.fc = nn.Sequential()  # stub for MVNetwork compat

        # Register normalisation as buffers so they move with .cuda()
        self.register_buffer("norm_mean", torch.tensor(self.MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("norm_std", torch.tensor(self.STD).view(1, 3, 1, 1, 1))

        cfg = _make_b16_cfg(num_frames=num_frames, drop_path=drop_path)

        # Import after sys.path is set
        import tadaconv.models.module_zoo.branches  # trigger BRANCH_REGISTRY
        from tadaconv.models.base.backbone import VisionTransformer

        self._vit = VisionTransformer(cfg)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            print(f"[TAdaFormer] WARNING: checkpoint not found at {checkpoint_path}")

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu")

        # TAdaConv checkpoints may nest state dict under various keys
        sd = ckpt
        for key in ("model_state", "state_dict", "model"):
            if key in ckpt:
                sd = ckpt[key]
                break

        # Strip "backbone." prefix if present
        sd = {k.replace("backbone.", ""): v for k, v in sd.items()}

        # Drop head weights — we don't use them
        sd = {
            k: v
            for k, v in sd.items()
            if not k.startswith("head.") and not k.startswith("proj")
        }

        missing, unexpected = self._vit.load_state_dict(sd, strict=False)
        print(f"[TAdaFormer] Loaded {path}")
        if missing:
            print(
                f"  Missing ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}"
            )
        if unexpected:
            print(
                f"  Unexpected ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}"
            )

    def _renormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert from MViT/ImageNet normalisation (mean=[0.45,0.45,0.45],
        std=[0.225,0.225,0.225]) to CLIP normalisation used by TAdaFormer.
        Input x is already normalised by the MViT transform in dataset.py.
        We undo that and apply CLIP stats instead.
        """
        MVIT_MEAN = torch.tensor([0.45, 0.45, 0.45], device=x.device).view(
            1, 3, 1, 1, 1
        )
        MVIT_STD = torch.tensor([0.225, 0.225, 0.225], device=x.device).view(
            1, 3, 1, 1, 1
        )
        x = x * MVIT_STD + MVIT_MEAN  # undo MViT norm → [0,1]
        x = (x - self.norm_mean) / self.norm_std  # apply CLIP norm
        return x

    def forward(self, x: torch.Tensor, return_tokens: bool = True) -> torch.Tensor:
        # x: [B, C, T, H, W]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        B, C, T, H, W = x.shape
        assert C == 3

        # Resample frames if needed
        if T != self.num_frames:
            x = F.interpolate(
                x, size=(self.num_frames, H, W), mode="trilinear", align_corners=False
            )

        if self.apply_renormalize:
            x = self._renormalize(x)

        if return_tokens:
            # Returns CLS token at each temporal step: [B, T', 768]
            # T' = num_frames // TUBLET_STRIDE = 16 // 2 = 8
            raw = self._vit.forward_wo_head(x)  # [B*T', patches+1, 768]
            T_out = self.num_frames // 2
            tokens = raw[:, 0, :].reshape(B, T_out, self.feat_dim)  # [B, T', 768]
            tokens = self._vit.ln_post(tokens)  # apply layer norm
            return tokens  # [B, 8, 768]

        out = self._vit(x)  # [B, 768]
        return out
