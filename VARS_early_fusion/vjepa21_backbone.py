# vjepa21_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class VJEPA21Backbone(nn.Module):
    """
    V-JEPA 2.1 ViT-B/16 backbone wrapper for MVNetwork.

    Pretrained on 1M hours internet video via JEPA objective.
    Input : [B, C, T, H, W] — standard MVNetwork format
    Output: [B, T', 768]    — temporal tokens for TransformerAggregate
    """

    def __init__(self, num_frames=16):
        super().__init__()
        print("[VJEPA21] Loading ViT-B/16 from torch.hub...")
        self.encoder = torch.hub.load(
            "facebookresearch/vjepa2",
            "vjepa2_1_vit_base_384",
            trust_repo=True,
        )
        self.feat_dim = 768
        self.num_frames = num_frames
        self.fc = nn.Sequential()  # dummy — matches MVNetwork interface
        print("[VJEPA21] Ready.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MVNetwork passes [B, C, T, H, W]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, C, T, H, W = x.shape

        # Resize to 384 (pretrained resolution)
        if H != 384 or W != 384:
            x = (
                F.interpolate(
                    x.view(B * C, T, H, W).unsqueeze(0),  # hack for 5D
                    size=(T, 384, 384),
                    mode="trilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .view(B, C, T, 384, 384)
            )

        # VJEPA expects [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Forward through encoder — returns patch tokens
        tokens = self.encoder(x)  # shape TBD — likely [B, N, 768]

        # N = T' * H' * W' spatial-temporal tokens
        # Need to reshape to [B, T', 768] for TransformerAggregate
        # With T=16, tubelet_size=2 → T'=8
        # With 384px, patch_size=16 → H'=W'=24
        # So N = 8 * 24 * 24 = 4608 tokens
        # Mean-pool spatial dims to get [B, T', 768]
        T_prime = T // 2  # tubelet_size=2
        N = tokens.shape[1]
        spatial_tokens = N // T_prime
        tokens = tokens.view(B, T_prime, spatial_tokens, 768)
        tokens = tokens.mean(dim=2)  # [B, T', 768]

        return tokens
