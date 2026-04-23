import torch
import torch.nn as nn
import torch.nn.functional as F
from mvaggregate import MVAggregate
from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights,
    mc3_18,
    MC3_18_Weights,
    r2plus1d_18,
    R2Plus1D_18_Weights,
    s3d,
    S3D_Weights,
    mvit_v2_s,
    MViT_V2_S_Weights,
    mvit_v1_b,
    MViT_V1_B_Weights,
)


# ---------------------------------------------------------------------------
# HuggingFace VideoMAE registry  (key → (hf_model_id, hidden_size))
# ---------------------------------------------------------------------------
HF_VIDEOMAE_REGISTRY = {
    "videomae2_base": ("OpenGVLab/VideoMAEv2-Base", 768),
    "videomae2_large": ("OpenGVLab/VideoMAEv2-Large", 1024),
    "videomae2_huge": ("OpenGVLab/VideoMAEv2-Huge", 1280),
    "videomae2_giant": ("OpenGVLab/VideoMAEv2-giant", 1408),
    "videomae_base": ("MCG-NJU/videomae-base", 768),
    "videomae_large": ("MCG-NJU/videomae-large", 1024),
}

_VIDEOMAE_V2_KEYS = {
    "videomae2_base",
    "videomae2_large",
    "videomae2_huge",
    "videomae2_giant",
}


# ---------------------------------------------------------------------------
# MViT-v2-S backbone — temporal token extraction (used by MVNetwork)
# ---------------------------------------------------------------------------


class MViTv2SBackbone(nn.Module):
    """
    MViT-v2-S with pre-head temporal token extraction.

    Input  : (B, C, T, H, W)  — resampled to 16 frames if T != 16
    Output : (B, T'=8, 768)   — spatial mean-pool over H'=7, W'=7
    """

    def __init__(self):
        super().__init__()
        base = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
        self.feat_dim = 768
        self.fc = nn.Sequential()
        self._last_thw = (8, 7, 7)
        self._norm_output = None

        def _thw_hook(*args):
            out = args[2]
            if isinstance(out, tuple) and len(out) == 2:
                _, thw = out
                if isinstance(thw, (tuple, list)) and len(thw) == 3:
                    self._last_thw = tuple(int(d) for d in thw)

        def _norm_hook(*args):
            self._norm_output = args[2]

        base.blocks[-1].register_forward_hook(_thw_hook)
        base.norm.register_forward_hook(_norm_hook)
        self._base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, C, T, H, W = x.shape
        assert C == 3, f"Expected 3 channels, got {x.shape}"
        if T != 16:
            x = F.interpolate(x, size=(16, H, W), mode="trilinear", align_corners=False)

        self._base(x)
        normed = self._norm_output

        Tp, Hp, Wp = self._last_thw
        N_patch = Tp * Hp * Wp
        patch_tokens = normed[:, 1:] if normed.shape[1] == N_patch + 1 else normed
        return patch_tokens.view(B, Tp, Hp, Wp, self.feat_dim).mean(
            dim=(2, 3)
        )  # [B, T', 768]


# ---------------------------------------------------------------------------
# VideoMAEv2 backbone (OpenGVLab)
# ---------------------------------------------------------------------------


class VideoMAEv2Backbone(nn.Module):
    def __init__(self, hf_model_id: str, hidden_size: int):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError:
            raise ImportError("pip install transformers")

        config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            hf_model_id, config=config, trust_remote_code=True
        )
        self.backbone.model.with_cp = True
        self.feat_dim = hidden_size
        self.pretrained_frames = 16
        self.fc = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, C, T, H, W = x.shape
        assert C == 3, f"Expected 3 channels, got shape {x.shape}"
        if T != self.pretrained_frames:
            x = F.interpolate(
                x,
                size=(self.pretrained_frames, H, W),
                mode="trilinear",
                align_corners=False,
            )
        return self.backbone(x)


# ---------------------------------------------------------------------------
# VideoMAEv1 backbone (MCG-NJU)
# ---------------------------------------------------------------------------


class VideoMAEv1Backbone(nn.Module):
    def __init__(self, hf_model_id: str, hidden_size: int):
        super().__init__()
        try:
            from transformers import VideoMAEModel
        except ImportError:
            raise ImportError("pip install transformers")

        self.backbone = VideoMAEModel.from_pretrained(hf_model_id)
        self.feat_dim = hidden_size
        self.pretrained_frames: int = self.backbone.config.num_frames
        self.fc = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        if T != self.pretrained_frames:
            x = F.interpolate(
                x,
                size=(self.pretrained_frames, H, W),
                mode="trilinear",
                align_corners=False,
            )
        pixel_values = x.permute(0, 2, 1, 3, 4).contiguous()
        out = self.backbone(pixel_values=pixel_values)
        return out.last_hidden_state.mean(dim=1)


# ---------------------------------------------------------------------------
# Standard multi-view wrapper (unchanged from VARS_model_v2)
# ---------------------------------------------------------------------------


class MVNetwork(torch.nn.Module):
    """
    Backbone + multi-view aggregation.

    Forward returns:
      (ordinal_severity_logits [B,3], action_logits [B,8],
       contact_logit [B], bodypart_logit [B],
       try_to_play_logit [B], handball_logit [B], attention)
    """

    def __init__(
        self,
        net_name: str = "mvit_v2_s",
        agr_type: str = "transformer",
        lifting_net: nn.Module = nn.Sequential(),
    ):
        super().__init__()
        self.net_name = net_name
        self.agr_type = agr_type
        self.feat_dim = 512

        if net_name in _VIDEOMAE_V2_KEYS:
            hf_id, hidden_size = HF_VIDEOMAE_REGISTRY[net_name]
            network = VideoMAEv2Backbone(hf_model_id=hf_id, hidden_size=hidden_size)
            self.feat_dim = hidden_size

        elif net_name in HF_VIDEOMAE_REGISTRY:
            hf_id, hidden_size = HF_VIDEOMAE_REGISTRY[net_name]
            network = VideoMAEv1Backbone(hf_model_id=hf_id, hidden_size=hidden_size)
            self.feat_dim = hidden_size

        elif net_name == "r3d_18":
            network = r3d_18(weights=R3D_18_Weights.DEFAULT)

        elif net_name == "s3d":
            network = s3d(weights=S3D_Weights.DEFAULT)
            self.feat_dim = 400

        elif net_name == "mc3_18":
            network = mc3_18(weights=MC3_18_Weights.DEFAULT)

        elif net_name == "r2plus1d_18":
            network = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        elif net_name == "mvit_v2_s":
            network = MViTv2SBackbone()
            self.feat_dim = 768

        elif net_name == "mvit_v1_b":
            network = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)
            self.feat_dim = 768

        else:
            print(
                f"Warning: unknown backbone '{net_name}', falling back to r2plus1d_18"
            )
            network = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        network.fc = nn.Sequential()

        self.mvnetwork = MVAggregate(
            model=network,
            agr_type=self.agr_type,
            feat_dim=self.feat_dim,
            lifting_net=lifting_net,
        )

    def forward(self, mvimages: torch.Tensor):
        return self.mvnetwork(mvimages)


# ---------------------------------------------------------------------------
# Early fusion backbone: single MViT-v2-S over all views interleaved in time
# ---------------------------------------------------------------------------


class EarlyFusionMViT(nn.Module):
    """
    MViT-v2-S operating on a temporally-fused multi-view clip.

    Input : [B, C, T*V, H, W] — views interleaved along the time axis
    Output: [B, 768]           — globally pooled representation

    A forward hook on base.head[0] (AdaptiveAvgPool3d) captures the
    [B, 768, 1, 1, 1] pooled tensor before the classification linear.
    """

    def __init__(self, num_views: int = 5, T_per_view: int = 16):
        super().__init__()
        self.num_views = num_views
        self.T_fused = num_views * T_per_view
        self.feat_dim = 768
        self._pooled_output = None

        base = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)

        # Find AdaptiveAvgPool3d wherever it sits in the head
        pool_layer = None
        for module in base.head.modules():
            if isinstance(module, nn.AdaptiveAvgPool3d):
                pool_layer = module
                break

        assert (
            pool_layer is not None
        ), f"Could not find AdaptiveAvgPool3d in base.head. Head structure: {base.head}"

        def _pool_hook(module, input, output):
            self._pooled_output = output

        pool_layer.register_forward_hook(_pool_hook)
        self._base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T*V, H, W]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        self._base(x)
        # _pooled_output: [B, 768, 1, 1, 1]
        return self._pooled_output.flatten(1)  # [B, 768]


# ---------------------------------------------------------------------------
# Early fusion network: single backbone + task heads, no MVAggregate
# ---------------------------------------------------------------------------


class EarlyFusionNetwork(nn.Module):
    """
    Early fusion: one MViT-v2-S backbone sees all V views interleaved in time.

    Input : fused_clip [B, C, T*V, H, W]
    Output: same 7-tuple as MVNetwork (attention is always None)
    """

    def __init__(self, num_views: int = 5, T_per_view: int = 16):
        super().__init__()
        self.backbone = EarlyFusionMViT(num_views, T_per_view)
        feat_dim = 768

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.fc_ordinal_severity = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, 3),
        )
        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 8),
        )
        self.fc_contact = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.fc_bodypart = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.fc_try_to_play = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1)
        )
        self.fc_handball = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))

    def forward(self, fused_clip: torch.Tensor):
        # fused_clip: [B, C, T*V, H, W]
        feat = self.backbone(fused_clip)  # [B, 768]
        inter = self.inter(feat)
        return (
            self.fc_ordinal_severity(inter),
            self.fc_action(inter),
            self.fc_contact(inter).squeeze(-1),
            self.fc_bodypart(inter).squeeze(-1),
            self.fc_try_to_play(inter).squeeze(-1),
            self.fc_handball(inter).squeeze(-1),
            None,  # no attention weights
        )
