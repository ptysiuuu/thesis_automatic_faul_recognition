# vjepa21_backbone.py
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

VJEPA2_REPO = "/net/tscratch/people/plgaszos/sn-mvfoul/vjepa2"
VJEPA21_CKPT = "/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion/checkpoints/vjepa21_vitb_384.pt"


class VJEPA21Backbone(nn.Module):
    """
    V-JEPA 2.1 ViT-B/16 backbone for MVNetwork.

    Pretraining: JEPA objective on 1M hours internet video
    Input : [B, C, T, H, W]  — MVNetwork format (C-first)
    Output: [B, T', 768]     — temporal tokens for TransformerAggregate

    Token structure (16f, 384px, patch=16, tubelet=2):
      T' = 16 / 2 = 8  temporal positions
      H' = W' = 384 / 16 = 24  spatial positions
      N  = 8 * 24 * 24 = 4608 tokens total
    Spatial mean-pool → [B, 8, 768]
    """

    def __init__(self, num_frames: int = 16, checkpoint_path: str = VJEPA21_CKPT):
        super().__init__()
        if VJEPA2_REPO not in sys.path:
            sys.path.insert(0, VJEPA2_REPO)

        print(f"[VJEPA21] Loading ViT-B/16 from {checkpoint_path}...")
        result = torch.hub.load(
            VJEPA2_REPO,
            "vjepa2_1_vit_base_384",
            source="local",
            trust_repo=True,
        )
        self.encoder = result[0] if isinstance(result, (tuple, list)) else result
        self.feat_dim = 768
        self.num_frames = num_frames
        self.tubelet_size = 2
        self.patch_size = 16
        self.img_size = 384
        self.fc = nn.Sequential()  # dummy — MVNetwork interface

        n_params = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        print(f"[VJEPA21] Ready. {n_params:.0f}M params.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        # Resize spatial dims to 384 if needed
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(
                x,
                size=(T, self.img_size, self.img_size),
                mode="trilinear",
                align_corners=False,
            )

        # VJEPA takes [B, C, T, H, W] directly
        result = self.encoder(x)
        tokens = (
            result[0] if isinstance(result, (tuple, list)) else result
        )  # [B, N, 768]

        # Reshape: N = T' * H' * W' → mean-pool spatial → [B, T', 768]
        T_prime = T // self.tubelet_size
        H_prime = self.img_size // self.patch_size
        N_expected = T_prime * H_prime * H_prime

        if tokens.shape[1] != N_expected:
            # Fallback: mean-pool all tokens and expand to T'
            tokens = tokens.mean(dim=1, keepdim=True).expand(-1, T_prime, -1)
        else:
            tokens = tokens.view(B, T_prime, H_prime * H_prime, 768)
            tokens = tokens.mean(dim=2)  # [B, T', 768]

        return tokens
