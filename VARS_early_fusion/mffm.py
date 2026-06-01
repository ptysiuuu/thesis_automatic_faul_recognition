import torch
import torch.nn as nn


class _MFFMLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm_self = nn.LayerNorm(dim)
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        visual_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # visual: [B, N, D], text: [B, M, D]
        # visual_pad_mask: [B, N] where True marks padded tokens
        v_norm = self.norm_self(visual)
        v_attn, _ = self.self_attn(
            v_norm,
            v_norm,
            v_norm,
            key_padding_mask=visual_pad_mask,
            need_weights=False,
        )
        visual = visual + v_attn

        q = self.norm_cross_q(visual)
        kv = self.norm_cross_kv(text)
        v_cross, _ = self.cross_attn(q, kv, kv, need_weights=False)
        visual = visual + v_cross

        visual = visual + self.ffn(self.norm_ffn(visual))
        return visual


class MultiModalFeatureFusionModule(nn.Module):
    """
    Decoder-style fusion between visual and text features.

    visual: [B, N, D]
    text:   [B, M, D]
    output: [B, N, D]
    """

    def __init__(self, dim: int = 768, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            [_MFFMLayer(dim=dim, num_heads=num_heads) for _ in range(num_layers)]
        )

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        visual_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if visual.dim() != 3 or text.dim() != 3:
            raise ValueError("Expected visual/text with shape [B, N, D]")
        if visual.shape[-1] != self.dim or text.shape[-1] != self.dim:
            raise ValueError("visual/text last dimension must match dim")
        if visual_pad_mask is not None:
            if visual_pad_mask.dim() != 2 or visual_pad_mask.shape[0] != visual.shape[0]:
                raise ValueError("visual_pad_mask must be [B, N]")

        x = visual
        for layer in self.layers:
            x = layer(x, text, visual_pad_mask=visual_pad_mask)
        return x
