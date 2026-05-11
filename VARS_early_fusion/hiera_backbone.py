"""
hiera_backbone.py
=================
Hiera video models as drop-in backbones for MVAggregate.

Input  : [B, C, T, H, W]
Output : [B, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

HIERA_VIDEO_MODELS = {
    "hiera_base_16x224": "mae_k400_ft_k400",
    "hiera_base_plus_16x224": "mae_k400_ft_k400",
    "hiera_large_16x224": "mae_k400_ft_k400",
    "hiera_huge_16x224": "mae_k400_ft_k400",
}


class HieraBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "hiera_base_16x224",
        checkpoint: str = None,
        pretrained: bool = True,
        num_frames: int = 16,
    ):
        super().__init__()
        if model_name not in HIERA_VIDEO_MODELS:
            raise ValueError(
                f"Unknown Hiera model '{model_name}'. Supported: {sorted(HIERA_VIDEO_MODELS)}"
            )
        if checkpoint is None:
            checkpoint = HIERA_VIDEO_MODELS[model_name]

        try:
            import hiera
        except ImportError as exc:
            raise ImportError(
                "hiera-transformer is required for Hiera backbones. "
                "Install it with `pip install hiera-transformer`."
            ) from exc

        factory = getattr(hiera, model_name, None)
        if factory is None:
            raise ValueError(f"Hiera factory '{model_name}' not found in hiera.")

        model = factory(pretrained=pretrained, checkpoint=checkpoint)
        feat_dim = None
        if hasattr(model, "head") and hasattr(model.head, "projection"):
            feat_dim = model.head.projection.in_features
        if hasattr(model, "head"):
            model.head = nn.Identity()
        if feat_dim is None and hasattr(model, "norm"):
            norm_shape = getattr(model.norm, "normalized_shape", None)
            if norm_shape:
                feat_dim = int(norm_shape[0])
        self.feat_dim = feat_dim or 768

        self._model = model
        self.num_frames = num_frames
        self.fc = nn.Sequential()  # stub for MVNetwork compat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, C, T, H, W = x.shape
        assert C == 3
        if T != self.num_frames:
            x = F.interpolate(
                x, size=(self.num_frames, H, W), mode="trilinear", align_corners=False
            )
        return self._model(x)
