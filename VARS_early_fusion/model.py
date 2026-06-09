import torch
import torch.nn as nn
import torch.nn.functional as F
from mvaggregate import MVAggregate
from mffm import MultiModalFeatureFusionModule
from task_decoder import TaskQueryDecoder
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
from backbone_attention_pool import (
    MViTv2SBackboneAttnPool,
    EarlyFusionMViTAttnPool,
    VideoMAEv2BackboneAttnPool,
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

HIERA_MODELS = {
    "hiera_base_16x224",
    "hiera_base_plus_16x224",
    "hiera_large_16x224",
    "hiera_huge_16x224",
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
             try_to_play_logit [B], handball_logit [B], attention,
             clip_proj [B,512] if enabled)
    """

    def __init__(
        self,
        net_name: str = "mvit_v2_s",
        agr_type: str = "transformer",
        lifting_net: nn.Module = nn.Sequential(),
        graph_topology: str = "structured",
        cascade_severity: bool = False,
        use_text_bridge: bool = False,
        clip_embeddings_path: str = "",
        use_mffm: bool = False,
        use_decoder: bool = False,
    ):
        super().__init__()
        self.net_name = net_name
        self.agr_type = agr_type
        self.graph_topology = graph_topology
        self.cascade_severity = cascade_severity
        self.feat_dim = 512
        self._tokens_per_view = None

        if net_name in _VIDEOMAE_V2_KEYS:
            hf_id, hidden_size = HF_VIDEOMAE_REGISTRY[net_name]
            network = VideoMAEv2Backbone(hf_model_id=hf_id, hidden_size=hidden_size)

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
            network = MViTv2SBackboneAttnPool()
            self.feat_dim = 768

        elif net_name == "mvit_v1_b":
            network = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)
            self.feat_dim = 768

        elif net_name in HIERA_MODELS:
            from hiera_backbone import HieraBackbone

            network = HieraBackbone(model_name=net_name)
            self.feat_dim = network.feat_dim
        elif net_name == "tadaformer_b16":
            from tadaformer_backbone import TAdaFormerBackbone

            network = TAdaFormerBackbone(
                checkpoint_path="/net/tscratch/people/plgaszos/sn-mvfoul/checkpoints/tadaformer_b16_k710.pth",
                num_frames=16,
                drop_path=0.1,
                apply_renormalize=False,
            )
            self.feat_dim = 768

        elif net_name == "aim_vitb16":
            from aim_backbone import AIMBackbone

            network = AIMBackbone(
                checkpoint_path="/net/tscratch/people/plgaszos/sn-mvfoul/checkpoints/vit_b_clip_16frame_k400.pth"
            )
            self.feat_dim = 768
            self._tokens_per_view = 16

        elif net_name == "vjepa21_vitb":
            from vjepa21_backbone import VJEPA21Backbone

            network = VJEPA21Backbone(num_frames=16)
            self.feat_dim = 768
            self._tokens_per_view = getattr(network, "tokens_per_view", None)

        elif net_name == "intern_video2_dist_b":
            from internvideo2_backbone import InternVideo2DistBBackbone

            network = InternVideo2DistBBackbone(num_frames=8)
            self.feat_dim = 768
            # _tokens_per_view stays None → T_max=8 default in TransformerAggregate

        elif net_name == "vjepa2_vitl":
            from vjepa21_backbone import VJEPA2ViTLBackbone

            network = VJEPA2ViTLBackbone(num_frames=16)
            self.feat_dim = network.feat_dim          # 1024, not hardcoded
            self._tokens_per_view = getattr(network, "tokens_per_view", None)  # 72
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
            graph_topology=self.graph_topology,
            cascade_severity=self.cascade_severity,
            use_text_bridge=use_text_bridge,
            clip_embeddings_path=clip_embeddings_path,
            T_max=self._tokens_per_view,
        )

        self.mffm = None
        if use_mffm:
            if self.feat_dim != 768:
                raise ValueError("MFFM requires 768-dim features")
            self.mffm = MultiModalFeatureFusionModule(dim=768, num_layers=2)

        self.decoder = None
        if use_decoder:
            if self.feat_dim % 8 == 0:
                num_heads = 8
            elif self.feat_dim % 4 == 0:
                num_heads = 4
            else:
                num_heads = 1
            self.decoder = TaskQueryDecoder(
                dim=self.feat_dim,
                num_layers=2,
                num_heads=num_heads,
                dropout=0.1,
            )

    def forward(self, mvimages: torch.Tensor, text_features: torch.Tensor = None):
        if self.decoder is not None:
            agg = self.mvnetwork.aggregation_model
            if not hasattr(agg, "get_tokens"):
                raise ValueError(
                    "Decoder requires aggregation_model.get_tokens; use pooling_type 'transformer'."
                )

            token_seq, token_mask, attention = agg.get_tokens(mvimages)
            if self.mffm is not None and text_features is not None:
                text = text_features.unsqueeze(1)
                token_seq = self.mffm(
                    token_seq, text, visual_pad_mask=token_mask
                )

            q_act, q_sev = self.decoder(token_seq, token_mask)

            pred_action = self.mvnetwork.fc_action(q_act)
            if self.mvnetwork.cascade_severity:
                sev_in = torch.cat([q_sev, pred_action.detach()], dim=-1)
            else:
                sev_in = q_sev
            pred_ordinal_severity = self.mvnetwork.fc_ordinal_severity(sev_in)

            pred_contact = self.mvnetwork.fc_contact(q_act).squeeze(-1)
            pred_bodypart = self.mvnetwork.fc_bodypart(q_act).squeeze(-1)
            pred_try_to_play = self.mvnetwork.fc_try_to_play(q_act).squeeze(-1)
            pred_handball = self.mvnetwork.fc_handball(q_act).squeeze(-1)

            clip_proj = None
            if self.mvnetwork.clip_proj is not None:
                clip_proj = F.normalize(self.mvnetwork.clip_proj(q_act), dim=-1)

            outputs = (
                pred_ordinal_severity,
                pred_action,
                pred_contact,
                pred_bodypart,
                pred_try_to_play,
                pred_handball,
                attention,
            )

            if clip_proj is None:
                return outputs
            return outputs + (clip_proj,)

        if self.mffm is None or text_features is None:
            return self.mvnetwork(mvimages)

        agg = self.mvnetwork.aggregation_model
        text = text_features.unsqueeze(1)

        if hasattr(agg, "get_tokens"):
            token_seq, token_mask, attention = agg.get_tokens(mvimages)
            fused_tokens = self.mffm(token_seq, text, visual_pad_mask=token_mask)

            if token_mask is not None:
                fused_tokens = fused_tokens.masked_fill(
                    token_mask.unsqueeze(-1), 0.0
                )
                valid_counts = (
                    (~token_mask).float().sum(dim=1, keepdim=True).clamp(min=1.0)
                )
                fused = fused_tokens.sum(dim=1) / valid_counts
            else:
                fused = fused_tokens.mean(dim=1)
        else:
            pooled_view, attention = agg(mvimages)
            visual = pooled_view.unsqueeze(1)
            fused = self.mffm(visual, text).squeeze(1)

        inter = self.mvnetwork.inter(fused)
        clip_proj = None
        if self.mvnetwork.clip_proj is not None:
            clip_proj = F.normalize(self.mvnetwork.clip_proj(inter), dim=-1)

        pred_action = self.mvnetwork.fc_action(inter)
        if self.mvnetwork.cascade_severity:
            sev_in = torch.cat([inter, pred_action.detach()], dim=-1)
            pred_ordinal_severity = self.mvnetwork.fc_ordinal_severity(sev_in)
        else:
            pred_ordinal_severity = self.mvnetwork.fc_ordinal_severity(inter)

        pred_contact = self.mvnetwork.fc_contact(inter).squeeze(-1)
        pred_bodypart = self.mvnetwork.fc_bodypart(inter).squeeze(-1)
        pred_try_to_play = self.mvnetwork.fc_try_to_play(inter).squeeze(-1)
        pred_handball = self.mvnetwork.fc_handball(inter).squeeze(-1)

        outputs = (
            pred_ordinal_severity,
            pred_action,
            pred_contact,
            pred_bodypart,
            pred_try_to_play,
            pred_handball,
            attention,
        )

        if clip_proj is None:
            return outputs
        return outputs + (clip_proj,)


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

        def _norm_hook(module, input, output):
            # output: [B, 768] — already pooled by MViT before the head
            self._pooled_output = output

        base.norm.register_forward_hook(_norm_hook)
        self._base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, C, T, H, W = x.shape
        if T != 16:
            x = F.interpolate(x, size=(16, H, W), mode="trilinear", align_corners=False)
        self._base(x)
        # _pooled_output is [B, N, 768] — mean-pool tokens to get [B, 768]
        return self._pooled_output.mean(dim=1)


# ---------------------------------------------------------------------------
# Early fusion network: single backbone + task heads, no MVAggregate
# ---------------------------------------------------------------------------


class EarlyFusionNetwork(nn.Module):
    """
    Early fusion: one MViT-v2-S backbone sees all V views interleaved in time.

    Input : fused_clip [B, C, T*V, H, W]
    Output: same 7-tuple as MVNetwork (attention is always None)
    """

    def __init__(
        self, num_views: int = 5, T_per_view: int = 16, cascade_severity: bool = False
    ):
        super().__init__()
        self.backbone = EarlyFusionMViTAttnPool(num_views, T_per_view)
        self.cascade_severity = cascade_severity
        feat_dim = 768

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        sev_in = feat_dim + 8 if cascade_severity else feat_dim
        self.fc_ordinal_severity = nn.Sequential(
            nn.LayerNorm(sev_in),
            nn.Dropout(0.3),
            nn.Linear(sev_in, feat_dim // 2),
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

    def forward(self, fused_clip: torch.Tensor, text_features: torch.Tensor = None):
        # fused_clip: [B, C, T*V, H, W]
        feat = self.backbone(fused_clip)  # [B, 768]
        inter = self.inter(feat)
        pred_action = self.fc_action(inter)
        if self.cascade_severity:
            sev_in = torch.cat([inter, pred_action], dim=-1)
            pred_sev = self.fc_ordinal_severity(sev_in)
        else:
            pred_sev = self.fc_ordinal_severity(inter)
        return (
            pred_sev,
            pred_action,
            self.fc_contact(inter).squeeze(-1),
            self.fc_bodypart(inter).squeeze(-1),
            self.fc_try_to_play(inter).squeeze(-1),
            self.fc_handball(inter).squeeze(-1),
            None,  # no attention weights
        )
