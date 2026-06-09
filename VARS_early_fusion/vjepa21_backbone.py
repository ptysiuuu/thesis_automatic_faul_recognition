# vjepa21_backbone.py
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

VJEPA2_REPO = "/net/tscratch/people/plgaszos/sn-mvfoul/vjepa2"
VJEPA21_CKPT = "/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion/checkpoints/vjepa21_vitb_384.pt"
VJEPA2_VITL_CKPT = "/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion/checkpoints/vjepa2_vitl_384.pt"


class VJEPA21Backbone(nn.Module):
    """
    V-JEPA 2.1 ViT-B/16 backbone for MVNetwork.

    Pretraining: JEPA objective on 1M hours internet video
    Input : [B, C, T, H, W]  — MVNetwork format (C-first)
    Output: [B, T', 768]     — spatiotemporal tokens for TransformerAggregate

    Token structure (16f, 384px, patch=16, tubelet=2):
      T' = 16 / 2 = 8  temporal positions
      H' = W' = 384 / 16 = 24  spatial positions
      N  = 8 * 24 * 24 = 4608 tokens total
    Spatial pool to 3x3 grid → 8 * 9 = 72 tokens per view
    """

    def __init__(
        self,
        num_frames: int = 16,
        checkpoint_path: str = VJEPA21_CKPT,
        use_adapter: bool = True,
    ):
        super().__init__()
        if VJEPA2_REPO not in sys.path:
            sys.path.insert(0, VJEPA2_REPO)

        print(f"[VJEPA21] Loading ViT-B/16 architecture...")
        result = torch.hub.load(
            VJEPA2_REPO,
            "vjepa2_1_vit_base_384",
            source="local",
            trust_repo=True,
        )
        self.encoder = result[0] if isinstance(result, (tuple, list)) else result

        # Explicitly load checkpoint — use ema_encoder (EMA weights = better quality)
        print(f"[VJEPA21] Loading weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # ema_encoder preferred over encoder (EMA = smoother, better for inference)
        state_dict = ckpt.get("ema_encoder") or ckpt.get("encoder")

        # Strip 'module.backbone.' prefix (DDP + backbone wrapper)
        state_dict = {
            k.replace("module.backbone.", ""): v for k, v in state_dict.items()
        }

        msg = self.encoder.load_state_dict(state_dict, strict=True)
        print(f"[VJEPA21] Weights loaded cleanly (strict=True).")

        self.feat_dim = 768
        self.num_frames = num_frames
        self.tubelet_size = 2
        self.patch_size = 16
        self.img_size = 384
        self.spatial_pool = (3, 3)
        self.tokens_per_view = 8 * self.spatial_pool[0] * self.spatial_pool[1]
        self.fc = nn.Sequential()

        self.jepa_adapter = None
        if use_adapter:
            self.jepa_adapter = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.GELU(),
                nn.Linear(self.feat_dim, self.feat_dim),
            )

        n_params = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[VJEPA21] Encoder frozen by default.")
        print(f"[VJEPA21] Ready. {n_params:.0f}M params.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        # Always use 16 frames and 384px — VJEPA pretrained config
        target_T = 16
        if T != target_T or H != self.img_size or W != self.img_size:
            x = F.interpolate(
                x,
                size=(target_T, self.img_size, self.img_size),
                mode="trilinear",
                align_corners=False,
            )

        tokens = self.encoder(x)  # [B, 4608, 768] — confirmed by probe

        # T'=8, H'=W'=24 always (tubelet=2, patch=16, img=384)
        tokens = tokens.view(B, 8, 24, 24, self.feat_dim)
        tokens = tokens.permute(0, 1, 4, 2, 3).reshape(B * 8, self.feat_dim, 24, 24)
        tokens = F.adaptive_max_pool2d(tokens, output_size=self.spatial_pool)
        tokens = tokens.view(
            B, 8, self.feat_dim, self.spatial_pool[0], self.spatial_pool[1]
        ).permute(0, 1, 3, 4, 2)
        tokens = tokens.reshape(
            B, 8 * self.spatial_pool[0] * self.spatial_pool[1], self.feat_dim
        )
        if self.jepa_adapter is not None:
            tokens = self.jepa_adapter(tokens)
        return tokens

    def get_layerwise_lr_groups(
        self, base_lr: float = 1e-5, decay: float = 0.65
    ) -> list:
        if not hasattr(self.encoder, "blocks"):
            return [{"params": self.encoder.parameters(), "lr": base_lr}]

        num_blocks = len(self.encoder.blocks)
        param_groups = []

        def _add_group(params, lr):
            params = [p for p in params if p.requires_grad]
            if params:
                param_groups.append({"params": params, "lr": lr})

        early_params = []
        for name in ("patch_embed", "pos_embed", "cls_token"):
            if not hasattr(self.encoder, name):
                continue
            obj = getattr(self.encoder, name)
            if isinstance(obj, nn.Parameter):
                early_params.append(obj)
            elif isinstance(obj, nn.Module):
                early_params.extend(list(obj.parameters()))

        early_lr = base_lr * (decay**num_blocks)
        _add_group(early_params, early_lr)

        for i, block in enumerate(self.encoder.blocks):
            lr = base_lr * (decay ** (num_blocks - 1 - i))
            _add_group(block.parameters(), lr)

        if hasattr(self.encoder, "norm"):
            _add_group(self.encoder.norm.parameters(), base_lr)

        return param_groups

    def train(self, mode=True):
        """Keep encoder in eval mode — disables dropout inside VJEPA."""
        super().train(mode)
        self.encoder.eval()
        return self


class VJEPA2ViTLBackbone(nn.Module):
    """
    V-JEPA 2.1 ViT-B/16 backbone for MVNetwork.

    Pretraining: JEPA objective on 1M hours internet video
    Input : [B, C, T, H, W]  — MVNetwork format (C-first)
    Output: [B, T', 768]     — spatiotemporal tokens for TransformerAggregate

    Token structure (16f, 384px, patch=16, tubelet=2):
      T' = 16 / 2 = 8  temporal positions
      H' = W' = 384 / 16 = 24  spatial positions
      N  = 8 * 24 * 24 = 4608 tokens total
    Spatial pool to 3x3 grid → 8 * 9 = 72 tokens per view
    """

    def __init__(
        self,
        num_frames: int = 16,
        checkpoint_path: str = VJEPA2_VITL_CKPT,
        use_adapter: bool = True,
    ):
        super().__init__()
        if VJEPA2_REPO not in sys.path:
            sys.path.insert(0, VJEPA2_REPO)

        print(f"[VJEPA21] Loading ViT-B/16 architecture...")
        result = torch.hub.load(
            VJEPA2_REPO,
            "vjepa2_1_vit_large_384",
            source="local",
            trust_repo=True,
        )
        self.encoder = result[0] if isinstance(result, (tuple, list)) else result

        # Explicitly load checkpoint — use ema_encoder (EMA weights = better quality)
        print(f"[VJEPA21] Loading weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # ema_encoder preferred over encoder (EMA = smoother, better for inference)
        state_dict = ckpt.get("ema_encoder") or ckpt.get("encoder")

        # Strip 'module.backbone.' prefix (DDP + backbone wrapper)
        state_dict = {
            k.replace("module.backbone.", ""): v for k, v in state_dict.items()
        }

        msg = self.encoder.load_state_dict(state_dict, strict=True)
        print(f"[VJEPA21] Weights loaded cleanly (strict=True).")

        self.feat_dim = 1024
        self.num_frames = num_frames
        self.tubelet_size = 2
        self.patch_size = 16
        self.img_size = 384
        self.spatial_pool = (3, 3)
        self.tokens_per_view = 8 * self.spatial_pool[0] * self.spatial_pool[1]
        self.fc = nn.Sequential()

        self.jepa_adapter = None
        if use_adapter:
            self.jepa_adapter = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.GELU(),
                nn.Linear(self.feat_dim, self.feat_dim),
            )

        n_params = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[VJEPA21] Encoder frozen by default.")
        print(f"[VJEPA21] Ready. {n_params:.0f}M params.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        # Always use 16 frames and 384px — VJEPA pretrained config
        target_T = 16
        if T != target_T or H != self.img_size or W != self.img_size:
            x = F.interpolate(
                x,
                size=(target_T, self.img_size, self.img_size),
                mode="trilinear",
                align_corners=False,
            )

        tokens = self.encoder(x)  # [B, 4608, 768] — confirmed by probe

        # T'=8, H'=W'=24 always (tubelet=2, patch=16, img=384)
        tokens = tokens.view(B, 8, 24, 24, self.feat_dim)
        tokens = tokens.permute(0, 1, 4, 2, 3).reshape(B * 8, self.feat_dim, 24, 24)
        tokens = F.adaptive_max_pool2d(tokens, output_size=self.spatial_pool)
        tokens = tokens.view(
            B, 8, self.feat_dim, self.spatial_pool[0], self.spatial_pool[1]
        ).permute(0, 1, 3, 4, 2)
        tokens = tokens.reshape(
            B, 8 * self.spatial_pool[0] * self.spatial_pool[1], self.feat_dim
        )
        if self.jepa_adapter is not None:
            tokens = self.jepa_adapter(tokens)
        return tokens

    def get_layerwise_lr_groups(
        self, base_lr: float = 1e-5, decay: float = 0.65
    ) -> list:
        if not hasattr(self.encoder, "blocks"):
            return [{"params": self.encoder.parameters(), "lr": base_lr}]

        num_blocks = len(self.encoder.blocks)
        param_groups = []

        def _add_group(params, lr):
            params = [p for p in params if p.requires_grad]
            if params:
                param_groups.append({"params": params, "lr": lr})

        early_params = []
        for name in ("patch_embed", "pos_embed", "cls_token"):
            if not hasattr(self.encoder, name):
                continue
            obj = getattr(self.encoder, name)
            if isinstance(obj, nn.Parameter):
                early_params.append(obj)
            elif isinstance(obj, nn.Module):
                early_params.extend(list(obj.parameters()))

        early_lr = base_lr * (decay**num_blocks)
        _add_group(early_params, early_lr)

        for i, block in enumerate(self.encoder.blocks):
            lr = base_lr * (decay ** (num_blocks - 1 - i))
            _add_group(block.parameters(), lr)

        if hasattr(self.encoder, "norm"):
            _add_group(self.encoder.norm.parameters(), base_lr)

        return param_groups

    def train(self, mode=True):
        """Keep encoder in eval mode — disables dropout inside VJEPA."""
        super().train(mode)
        self.encoder.eval()
        return self