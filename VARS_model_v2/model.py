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
# HuggingFace VideoMAE (v1 + v2) registry
# key  →  (hf_model_id, hidden_size)
# ---------------------------------------------------------------------------
HF_VIDEOMAE_REGISTRY = {
    # VideoMAEv2 — pretrained on UnlabeledHybrid (much stronger than v1 on Kinetics)
    'videomae2_base':   ('OpenGVLab/VideoMAEv2-base',   768),
    'videomae2_large':  ('OpenGVLab/VideoMAEv2-large',  1024),
    'videomae2_giant':  ('OpenGVLab/VideoMAEv2-giant',  1408),
    # VideoMAEv1 — pretrained on Kinetics-400
    'videomae_base':    ('MCG-NJU/videomae-base',       768),
    'videomae_large':   ('MCG-NJU/videomae-large',      1024),
}


class VideoMAEv2Backbone(nn.Module):
    """
    HuggingFace VideoMAEModel wrapper — drop-in replacement for torchvision backbones.

    Input  : (B, C, T, H, W)  — same layout that batch_tensor() produces
    Output : (B, hidden_size) — global feature via mean-pool of patch tokens

    Any-FPS support
    ---------------
    The pretrained model expects exactly `pretrained_frames` frames (16 for base/large).
    If the incoming clip has a different temporal length T, the tensor is trilinearly
    resampled to `pretrained_frames` before being fed to the model.  This lets you
    call the model with any sampling rate without retraining — quality degrades slightly
    vs fine-tuning at the target fps, but it removes a hard constraint on the pipeline.
    """

    def __init__(self, hf_model_id: str, hidden_size: int):
        super().__init__()
        try:
            from transformers import VideoMAEModel
        except ImportError:
            raise ImportError(
                "pip install transformers  — required for VideoMAEv2 backbone"
            )
        self.backbone = VideoMAEModel.from_pretrained(hf_model_id)
        self.feat_dim = hidden_size
        # Number of frames the model was pretrained on (usually 16)
        self.pretrained_frames: int = self.backbone.config.num_frames
        # Stub — MVNetwork sets network.fc = nn.Sequential() on every backbone;
        # this attribute keeps that line from raising AttributeError.
        self.fc = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # --- temporal resampling (any-fps) ---
        if T != self.pretrained_frames:
            x = F.interpolate(
                x,
                size=(self.pretrained_frames, H, W),
                mode='trilinear',
                align_corners=False,
            )

        # VideoMAE expects (B, T, C, H, W)
        pixel_values = x.permute(0, 2, 1, 3, 4).contiguous()

        # Forward — no tube masking at inference time
        out = self.backbone(pixel_values=pixel_values)

        # Mean-pool over patch tokens: (B, num_patches, hidden) → (B, hidden)
        return out.last_hidden_state.mean(dim=1)


# ---------------------------------------------------------------------------
# Main model wrapper
# ---------------------------------------------------------------------------

class MVNetwork(torch.nn.Module):
    """
    Backbone + multi-view aggregation.

    Supported backbones (--pre_model):
      torchvision : r3d_18 | mc3_18 | r2plus1d_18 | s3d | mvit_v2_s | mvit_v1_b
      HF VideoMAE : videomae_base | videomae_large
                    videomae2_base | videomae2_large | videomae2_giant

    Forward output:
      (ordinal_severity_logits [B,3],
       action_logits [B,8],
       contact_logit [B],
       bodypart_logit [B],
       try_to_play_logit [B],
       handball_logit [B],
       attention)
    """

    def __init__(self, net_name: str = 'mvit_v2_s', agr_type: str = 'transformer',
                 lifting_net: nn.Module = nn.Sequential()):
        super().__init__()
        self.net_name = net_name
        self.agr_type = agr_type
        self.feat_dim = 512  # default; overridden below

        if net_name in HF_VIDEOMAE_REGISTRY:
            hf_id, hidden_size = HF_VIDEOMAE_REGISTRY[net_name]
            network = VideoMAEv2Backbone(hf_model_id=hf_id, hidden_size=hidden_size)
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
            network = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
            self.feat_dim = 400

        elif net_name == "mvit_v1_b":
            network = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)
            self.feat_dim = 768

        else:
            print(f"Warning: unknown backbone '{net_name}', falling back to r2plus1d_18")
            network = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        # For torchvision models: strip the classification head so forward()
        # returns feature vectors.  For VideoMAEv2Backbone this is a no-op
        # (the stub fc attribute is overwritten with the same empty Sequential).
        network.fc = nn.Sequential()

        self.mvnetwork = MVAggregate(
            model=network,
            agr_type=self.agr_type,
            feat_dim=self.feat_dim,
            lifting_net=lifting_net,
        )

    def forward(self, mvimages: torch.Tensor):
        return self.mvnetwork(mvimages)
