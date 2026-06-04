import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_clip import ViT_CLIP

logger = logging.getLogger(__name__)


class AIMBackbone(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.feat_dim = 768
        self.fc = nn.Sequential()
        self._vit = ViT_CLIP(
            input_resolution=224,
            patch_size=16,
            width=768,
            layers=12,
            heads=12,
            num_frames=16,
            drop_path_rate=0.1,
            num_tadapter=1,
            adapter_scale=0.5,
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint
            for key in ("state_dict", "model", "model_state"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            state_dict = {
        		k[len("backbone."):] if k.startswith("backbone.") else k: v
        		for k, v in state_dict.items()
    		}

            missing, unexpected = self._vit.load_state_dict(state_dict, strict=False)
            if missing:
                logger.info("AIM missing keys: %s", missing)
            if unexpected:
                logger.info("AIM unexpected keys: %s", unexpected)
        else:
            logger.warning("AIM checkpoint not found at %s", checkpoint_path)

        for name, param in self._vit.named_parameters():
            if "Adapter" not in name and "temporal_embedding" not in name:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != 16:
            x = F.interpolate(x, size=(16, x.shape[3], x.shape[4]),
						mode='trilinear', align_corners=False)
        x = self._vit(x)
        x = x.squeeze(-1).squeeze(-1)
        return x.permute(0, 2, 1).contiguous()
