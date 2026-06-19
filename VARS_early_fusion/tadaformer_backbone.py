"""
tadaformer_backbone.py
======================
TAdaFormer-B/16 and L/14 as drop-in backbones for MVAggregate.

Input  : [B, C, T, H, W]
Output : [B, 768] or [B, 1024] depending on arch

Notes:
- Optionally renormalizes from MViT/ImageNet stats to CLIP stats
- num_frames=16, tublet_stride=2 → 8 temporal tokens
- forward() returns [B, D] via CLS token mean across temporal dim
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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


def _make_l14_cfg(num_frames=16, num_classes=0, drop_path=0.1):
    return _Cfg(
        {
            "VIDEO": {
                "BACKBONE": {
                    "META_ARCH": "VisionTransformer",
                    "INPUT_RES": 224,
                    "PATCH_SIZE": 14,
                    "TUBLET_SIZE": 3,
                    "TUBLET_STRIDE": 2,
                    "NUM_FEATURES": 1024,
                    "NUM_OUT_FEATURES": 1024,
                    "DEPTH": 24,
                    "NUM_HEADS": 16,
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


def _resize_positional_embedding(self, tgt_h: int, tgt_w: int):
    """Bicubic resize of spatial positional embedding after checkpoint load."""
    tgt_n = 1 + tgt_h * tgt_w
    for module in self._vit.modules():
        if not hasattr(module, "positional_embedding"):
            continue
        pe = module.positional_embedding          # nn.Parameter [N, D]
        if pe.shape[0] == tgt_n:
            return
        N, D = pe.shape
        src_hw = int(round((N - 1) ** 0.5))
        cls_pe   = pe[:1, :].detach()
        patch_pe = pe[1:, :].detach()
        patch_pe = patch_pe.reshape(1, src_hw, src_hw, D).permute(0, 3, 1, 2).float()
        patch_pe = F.interpolate(
            patch_pe, size=(tgt_h, tgt_w), mode="bicubic", align_corners=False
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(tgt_h * tgt_w, D).to(pe.dtype)
        module.positional_embedding = nn.Parameter(
            torch.cat([cls_pe, patch_pe], dim=0)
        )
        print(f"[TAdaFormer] Resized pos. embed: {N} → {tgt_n} ({src_hw}×{src_hw} → {tgt_h}×{tgt_w})")
        return

# ---------------------------------------------------------------------------
# Backbone wrapper
# ---------------------------------------------------------------------------


class TAdaFormerBackbone(nn.Module):
    """
    Wraps TAdaFormer-B/16 or L/14 for use as a drop-in backbone in MVAggregate.

    Args:
        checkpoint_path : path to downloaded .pyth checkpoint
        num_frames      : must match the checkpoint (16 for K400/K710 ckp)
        drop_path       : stochastic depth rate (0.1 for fine-tuning)
        arch            : "b16" (768-dim, 12 layers) or "l14" (1024-dim, 24 layers)
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
        arch: str = "b16",
        gradient_checkpointing: bool = False,
        spatial_size: tuple = None,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.apply_renormalize = apply_renormalize
        self.fc = nn.Sequential()  # stub for MVNetwork compat

        if arch == "l14":
            cfg = _make_l14_cfg(num_frames=num_frames, drop_path=drop_path)
            self.feat_dim = 1024
        else:
            cfg = _make_b16_cfg(num_frames=num_frames, drop_path=drop_path)
            self.feat_dim = 768

        self._arch = arch
        self._tgt_spatial = spatial_size  # (tgt_h, tgt_w) in patch units, e.g. (16, 28) for 224×392

        # Register normalisation as buffers so they move with .cuda()
        self.register_buffer("norm_mean", torch.tensor(self.MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("norm_std", torch.tensor(self.STD).view(1, 3, 1, 1, 1))

        # Import after sys.path is set
        import tadaconv.models.module_zoo.branches  # trigger BRANCH_REGISTRY
        from tadaconv.models.base.backbone import VisionTransformer

        self._vit = VisionTransformer(cfg)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            print(f"[TAdaFormer-{arch}] WARNING: checkpoint not found at {checkpoint_path}")

        if spatial_size is not None:                         # ← add these two lines
            self._resize_positional_embedding(*spatial_size)
        if gradient_checkpointing:
            for block in self._vit.layers:
                orig_fwd = block.forward
                block.forward = lambda x, fn=orig_fwd: checkpoint(
                    fn, x, use_reentrant=False
                )

    def _maybe_interpolate_temporal(self, sd: dict) -> dict:
        """
        Interpolate temporal positional embeddings to match self.num_frames.

        TAdaFormer stores temporal position embeddings (named *temporal*) with
        a temporal axis equal to the num_frames the checkpoint was trained on.
        When fine-tuning with a different frame count we linearly interpolate
        along that axis so the rest of the weights can load cleanly.

        Handles [1, T, D] and [T, D] layouts.
        """
        tgt_T = self.num_frames
        out = {}
        for name, tensor in sd.items():
            if "temporal" in name.lower():
                if tensor.dim() == 3 and 2 <= tensor.shape[1] <= 128:
                    src_T = tensor.shape[1]
                    if src_T != tgt_T:
                        # [1, src_T, D] → linear interp → [1, tgt_T, D]
                        emb = tensor.float().permute(0, 2, 1)  # [1, D, src_T]
                        emb = F.interpolate(
                            emb, size=tgt_T, mode="linear", align_corners=False
                        )
                        tensor = emb.permute(0, 2, 1).to(tensor.dtype)
                        print(
                            f"[TAdaFormer-{self._arch}] Interpolated '{name}': T={src_T} → T={tgt_T}"
                        )
                elif tensor.dim() == 2 and 2 <= tensor.shape[0] <= 128:
                    src_T = tensor.shape[0]
                    if src_T != tgt_T:
                        # [src_T, D] → linear interp → [tgt_T, D]
                        emb = tensor.float().unsqueeze(0).permute(0, 2, 1)  # [1, D, src_T]
                        emb = F.interpolate(
                            emb, size=tgt_T, mode="linear", align_corners=False
                        )
                        tensor = emb.permute(0, 2, 1).squeeze(0).to(tensor.dtype)
                        print(
                            f"[TAdaFormer-{self._arch}] Interpolated '{name}': T={src_T} → T={tgt_T}"
                        )
            out[name] = tensor
        return out

    def _maybe_interpolate_spatial(self, sd: dict, tgt_h: int, tgt_w: int) -> dict:
        """Bicubic interpolation of 2D spatial positional embeddings for non-square inputs."""
        out = {}
        for name, tensor in sd.items():
            if ("pos_embed" in name or "spatial_embed" in name) and tensor.dim() == 3:
                N = tensor.shape[1] - 1          # subtract CLS token
                src_hw = int(N ** 0.5)
                if src_hw * src_hw != N:
                    out[name] = tensor
                    continue                      # skip non-square grids
                if src_hw == tgt_h and src_hw == tgt_w:
                    out[name] = tensor
                    continue                      # already correct size

                cls_pe   = tensor[:, :1, :]      # [1, 1, D]
                patch_pe = tensor[:, 1:, :]       # [1, N, D]
                D = patch_pe.shape[-1]

                patch_pe = patch_pe.reshape(1, src_hw, src_hw, D)
                patch_pe = patch_pe.permute(0, 3, 1, 2).float()   # [1, D, H, W]
                patch_pe = F.interpolate(
                    patch_pe, size=(tgt_h, tgt_w),
                    mode="bicubic", align_corners=False
                ).to(tensor.dtype)
                patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, tgt_h * tgt_w, D)

                out[name] = torch.cat([cls_pe, patch_pe], dim=1)
                print(f"[TAdaFormer] Interpolated '{name}': {src_hw}×{src_hw} → {tgt_h}×{tgt_w}")
            else:
                out[name] = tensor
        return out

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

        # Interpolate temporal positional embeddings when num_frames != checkpoint default
        sd = self._maybe_interpolate_temporal(sd)
        # Phase 2: interpolate spatial PE for non-square inputs (e.g. 224×392)
        if self._tgt_spatial is not None:
            sd = self._maybe_interpolate_spatial(sd, *self._tgt_spatial)

        missing, unexpected = self._vit.load_state_dict(sd, strict=False)
        print(f"[TAdaFormer-{self._arch}] Loaded {path}")
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

    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
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
            # Returns CLS token at each temporal step: [B, T', D]
            # T' = num_frames // TUBLET_STRIDE = num_frames // 2
            raw = self._vit.forward_wo_head(x)  # [B*T', patches+1, D]
            T_out = self.num_frames // 2
            tokens = raw[:, 0, :].reshape(B, T_out, self.feat_dim)  # [B, T', D]
            tokens = self._vit.ln_post(tokens)  # apply layer norm
            return tokens  # [B, T', D]

        out = self._vit(x)  # [B, 768]
        return out
