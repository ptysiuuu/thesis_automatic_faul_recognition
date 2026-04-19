import torch
import torch.nn as nn
import torch.nn.functional as F
from mvaggregate import MVAggregate
from torchvision.models.video import (
    r3d_18, R3D_18_Weights,
    mc3_18, MC3_18_Weights,
    r2plus1d_18, R2Plus1D_18_Weights,
    s3d, S3D_Weights,
    mvit_v2_s, MViT_V2_S_Weights,
    mvit_v1_b, MViT_V1_B_Weights,
)


# ---------------------------------------------------------------------------
# HuggingFace VideoMAE registry  (key → (hf_model_id, hidden_size))
#
# v2 keys (videomae2_*) — OpenGVLab custom-code models, loaded with
#   AutoModel + trust_remote_code=True.  Forward: (B,C,T,H,W) → (B, D).
#
# v1 keys (videomae_*)  — MCG-NJU standard HF models, loaded with
#   VideoMAEModel.  Forward: (B,C,T,H,W) → permute → (B,T,C,H,W) →
#   mean-pool last_hidden_state → (B, D).
# ---------------------------------------------------------------------------
HF_VIDEOMAE_REGISTRY = {
    # VideoMAEv2 — pretrained on UnlabeledHybrid (case-sensitive HF IDs)
    'videomae2_base':   ('OpenGVLab/VideoMAEv2-Base',   768),
    'videomae2_large':  ('OpenGVLab/VideoMAEv2-Large',  1024),
    'videomae2_huge':   ('OpenGVLab/VideoMAEv2-Huge',   1280),
    'videomae2_giant':  ('OpenGVLab/VideoMAEv2-giant',  1408),
    # VideoMAEv1 — pretrained on Kinetics-400
    'videomae_base':    ('MCG-NJU/videomae-base',       768),
    'videomae_large':   ('MCG-NJU/videomae-large',      1024),
}

# Keys that use the custom OpenGVLab architecture (trust_remote_code required)
_VIDEOMAE_V2_KEYS = {'videomae2_base', 'videomae2_large', 'videomae2_huge', 'videomae2_giant'}


# ---------------------------------------------------------------------------
# VideoMAEv2 backbone (OpenGVLab — custom code)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MViT-v2-S backbone — temporal token extraction
# ---------------------------------------------------------------------------

class MViTv2SBackbone(nn.Module):
    """
    MViT-v2-S with pre-head temporal token extraction.

    Registers hooks on (a) the last transformer block to capture the final
    (T', H', W') spatial-temporal dimensions, and (b) the norm layer to
    capture the full [B, N, C] token sequence before the head pools it away.

    Input  : (B, C, T, H, W)  — resampled to 16 frames if T != 16
    Output : (B, T'=8, 768)   — spatial mean-pool over H'=7, W'=7
    """

    def __init__(self):
        super().__init__()
        base = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
        self.feat_dim = 768
        self.fc = nn.Sequential()      # stub for MVNetwork compatibility
        self._last_thw = (8, 7, 7)     # updated by hook; default for 16f/224²
        self._norm_output = None

        def _thw_hook(*args):
            out = args[2]
            if isinstance(out, tuple) and len(out) == 2:
                _, thw = out
                if isinstance(thw, (tuple, list)) and len(thw) == 3:
                    self._last_thw = tuple(int(d) for d in thw)

        def _norm_hook(*args):
            self._norm_output = args[2]    # [B, N or N+1, 768]

        base.blocks[-1].register_forward_hook(_thw_hook)
        base.norm.register_forward_hook(_norm_hook)
        self._base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, C, T, H, W = x.shape
        assert C == 3, f"Expected 3 channels, got {x.shape}"
        if T != 16:
            x = F.interpolate(x, size=(16, H, W), mode='trilinear', align_corners=False)

        self._base(x)                         # populate hooks (return value discarded)
        normed = self._norm_output            # [B, N or N+1, 768]

        Tp, Hp, Wp = self._last_thw           # (8, 7, 7)
        N_patch = Tp * Hp * Wp
        # Drop CLS token if present
        patch_tokens = normed[:, 1:] if normed.shape[1] == N_patch + 1 else normed
        # [B, N, 768] → [B, T', H', W', 768] → spatial mean-pool
        return patch_tokens.view(B, Tp, Hp, Wp, self.feat_dim).mean(dim=(2, 3))  # [B, T', 768]


class VideoMAEv2Backbone(nn.Module):
    """
    OpenGVLab VideoMAEv2 loaded via AutoModel + trust_remote_code=True.

    Input  : (B, C, T, H, W)  — torchvision layout from batch_tensor()
    Output : (B, hidden_size) — already globally pooled by the model's head

    Any-FPS: clips are trilinearly resampled to 16 frames if needed.
    """

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
        self.pretrained_frames = 16   # VideoMAEv2 is fixed at 16 frames
        self.fc = nn.Sequential()     # stub so MVNetwork can do network.fc = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       # If someone passed us (B, T, C, H, W) instead of (B, C, T, H, W)
       # Heuristic: channel dim should be 3 for RGB.
       if x.shape[1] != 3 and x.shape[2] == 3:
           x = x.permute(0, 2, 1, 3, 4).contiguous()
       B, C, T, H, W = x.shape
       assert C == 3, f"Expected 3 channels, got shape {x.shape}"
       if T != self.pretrained_frames:
           x = F.interpolate(x, size=(self.pretrained_frames, H, W),
                             mode='trilinear', align_corners=False)
       return self.backbone(x)


# ---------------------------------------------------------------------------
# VideoMAEv1 backbone (MCG-NJU — standard HF VideoMAEModel)
# ---------------------------------------------------------------------------

class VideoMAEv1Backbone(nn.Module):
    """
    Standard HF VideoMAEModel (MCG-NJU checkpoints, Kinetics-400).

    Input  : (B, C, T, H, W)
    Output : (B, hidden_size) — mean-pool of last_hidden_state patch tokens

    Any-FPS: clips are trilinearly resampled to pretrained_frames if needed.
    """

    def __init__(self, hf_model_id: str, hidden_size: int):
        super().__init__()
        try:
            from transformers import VideoMAEModel
        except ImportError:
            raise ImportError("pip install transformers")

        self.backbone = VideoMAEModel.from_pretrained(hf_model_id)
        self.feat_dim = hidden_size
        self.pretrained_frames: int = self.backbone.config.num_frames  # usually 16
        self.fc = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        if T != self.pretrained_frames:
            x = F.interpolate(
                x, size=(self.pretrained_frames, H, W),
                mode='trilinear', align_corners=False,
            )
        # VideoMAEModel expects (B, T, C, H, W)
        pixel_values = x.permute(0, 2, 1, 3, 4).contiguous()
        out = self.backbone(pixel_values=pixel_values)
        # Mean-pool patch tokens: (B, num_patches, D) → (B, D)
        return out.last_hidden_state.mean(dim=1)


# ---------------------------------------------------------------------------
# Main model wrapper
# ---------------------------------------------------------------------------

class MVNetwork(torch.nn.Module):
    """
    Backbone + multi-view aggregation.

    Supported --pre_model values:
      torchvision : r3d_18 | mc3_18 | r2plus1d_18 | s3d | mvit_v2_s | mvit_v1_b
      HF v1       : videomae_base | videomae_large
      HF v2       : videomae2_base | videomae2_large | videomae2_huge | videomae2_giant

    Forward returns:
      (ordinal_severity_logits [B,3], action_logits [B,8],
       contact_logit [B], bodypart_logit [B],
       try_to_play_logit [B], handball_logit [B], attention)
    """

    def __init__(self, net_name: str = 'mvit_v2_s', agr_type: str = 'transformer',
                 lifting_net: nn.Module = nn.Sequential()):
        super().__init__()
        self.net_name = net_name
        self.agr_type = agr_type
        self.feat_dim = 512  # default; overridden below

        if net_name in _VIDEOMAE_V2_KEYS:
            hf_id, hidden_size = HF_VIDEOMAE_REGISTRY[net_name]
            network = VideoMAEv2Backbone(hf_model_id=hf_id, hidden_size=hidden_size)
            self.feat_dim = hidden_size

        elif net_name in HF_VIDEOMAE_REGISTRY:          # v1 keys
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
            print(f"Warning: unknown backbone '{net_name}', falling back to r2plus1d_18")
            network = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        # Remove the classification FC from torchvision models.
        # For HF backbones this overwrites the stub attribute — harmless.
        network.fc = nn.Sequential()

        self.mvnetwork = MVAggregate(
            model=network,
            agr_type=self.agr_type,
            feat_dim=self.feat_dim,
            lifting_net=lifting_net,
        )

    def forward(self, mvimages: torch.Tensor):
        return self.mvnetwork(mvimages)
