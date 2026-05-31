# internvideo2_backbone.py
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

IV2_SRC  = "/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion/internvideo2_src"
IV2_CKPT = "/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion/checkpoints/internvideo2_dist_b14.pth"


class InternVideo2DistBBackbone(nn.Module):
    """
    InternVideo2-dist-B/14 feature extractor.

    Distilled from InternVideo2-Stage2-1B teacher (has CLIP alignment + VideoMAE motion).
    Input : [B, C, T, H, W]  —  MVNetwork format, C-first
    Output: [B, T'=8, 768]   —  temporal tokens for TransformerAggregate

    Token structure (8f, 224px, patch=14, tubelet=1):
        T' = 8 / 1 = 8   temporal positions
        H' = W' = 16      spatial positions per frame  (224 // 14)
        N  = 8*256+1=2049  tokens total (+ CLS); spatial mean-pooled to [B, 8, 768]
    """

    def __init__(self, checkpoint_path: str = IV2_CKPT, num_frames: int = 8):
        super().__init__()

        if IV2_SRC not in sys.path:
            sys.path.insert(0, IV2_SRC)

        from internvideo2 import internvideo2_base_patch14_224  # noqa

        print("[IV2-dist-B] Building architecture (embed_dim=768, depth=12)...")
        self.backbone = internvideo2_base_patch14_224(
            num_frames=num_frames,
            tubelet_size=1,
            use_flash_attn=True,
            use_fused_mlp=True,
            use_fused_rmsnorm=True,
            sep_pos_embed=False,
            num_classes=400,        # placeholder; head weights are never used
            use_checkpoint=False,
        )

        print(f"[IV2-dist-B] Loading weights from {checkpoint_path}...")
        raw = torch.load(checkpoint_path, map_location="cpu")
        # checkpoint may be wrapped in a dict
        state_dict = raw.get("model", raw.get("module", raw))
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        missing = [k for k in msg.missing_keys if "head" not in k]
        if missing:
            print(f"[IV2-dist-B] Unexpected missing keys: {missing[:5]}")
        print("[IV2-dist-B] Weights loaded.")

        self.feat_dim = 768
        self.num_frames = num_frames
        self.fc = nn.Sequential()   # required by model.py: network.fc = nn.Sequential()

        # Hook: capture last-block output BEFORE clip_projector
        self._tokens: torch.Tensor | None = None

        def _last_block_hook(module, inp, out):
            # With use_fused_rmsnorm=False blocks return a plain tensor
            self._tokens = out if isinstance(out, torch.Tensor) else out[0]

        self.backbone.blocks[-1].register_forward_hook(_last_block_hook)

        # Freeze all encoder parameters by default
        for p in self.backbone.parameters():
            p.requires_grad = False

        n = sum(p.numel() for p in self.backbone.parameters()) / 1e6
        print(f"[IV2-dist-B] Ready — {n:.0f}M params, frozen.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        # Resize if needed (safety guard; transform should already give 224px, 8f)
        if T != self.num_frames or H != 224 or W != 224:
            x = F.interpolate(
                x, size=(self.num_frames, 224, 224),
                mode="trilinear", align_corners=False,
            )

        self._tokens = None
        self.backbone(x)            # forward; hook fires on last block

        tokens = self._tokens       # [B, T'*H'*W' + 1, 768]
        if tokens is None:
            raise RuntimeError("[IV2-dist-B] Forward hook did not fire.")

        tokens = tokens[:, 1:]      # strip CLS → [B, 2048, 768]
        T_prime = self.num_frames   # 8
        L = (224 // 14) ** 2       # 256
        tokens = tokens.view(B, T_prime, L, self.feat_dim)
        tokens = tokens.mean(dim=2) # spatial mean-pool → [B, 8, 768]
        return tokens

    def train(self, mode: bool = True):
        """Keep backbone in eval mode (disables dropout) like vjepa21_backbone."""
        super().train(mode)
        self.backbone.eval()
        return self
