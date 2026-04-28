import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Core module
# =============================================================================


class AttentionPool1D(nn.Module):
    """
    Replaces mean pooling over a sequence of tokens.

    Input  : [B, N, D]   — N tokens, D-dimensional
    Output : [B, D]      — single weighted-sum vector

    The query is a learnable [1, 1, D] vector. Attention scores are
    computed as scaled dot-product between the query and all tokens,
    then softmaxed to get weights, then used to compute a weighted sum.

    Optionally accepts a padding mask [B, N] (True = ignore).
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.scale = feat_dim**-0.5
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x    : [B, N, D]
        mask : [B, N] bool, True = padded (ignored)
        """
        B, N, D = x.shape

        # Scores: [B, N]
        q = self.query.expand(B, -1, -1)  # [B, 1, D]
        scores = torch.bmm(q, x.transpose(1, 2)).squeeze(1) * self.scale  # [B, N]

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # [B, N]
        weights = torch.nan_to_num(weights, nan=0.0)

        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)  # [B, D]
        return pooled


# =============================================================================
# Modified backbone classes
# =============================================================================


class MViTv2SBackboneAttnPool(nn.Module):
    """
    MViT-v2-S with attention pooling instead of spatial mean pooling.

    Original: patch_tokens.view(...).mean(dim=(2, 3))  → [B, T', D]
              then in WeightedAggregate/TransformerAggregate: .mean(dim=2) → [B, D]

    This version:
      - spatial attn pool: [B, T'*H'*W', D] → [B, T', D]  (pool over H'*W' per timestep)
      - temporal attn pool: handled downstream by aggregator (unchanged)

    The spatial pool has its own query per timestep (shared across B).
    This is the minimal change — only spatial GAP is replaced.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

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

        # Attention pool over spatial tokens per timestep
        # One shared query for all T' timesteps (parameter-efficient)
        self.spatial_pool = AttentionPool1D(feat_dim=768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, C, T, H, W = x.shape
        assert C == 3
        if T != 16:
            x = F.interpolate(x, size=(16, H, W), mode="trilinear", align_corners=False)

        self._base(x)
        normed = self._norm_output  # [B, 1 + T'*H'*W', D] or [B, T'*H'*W', D]

        Tp, Hp, Wp = self._last_thw
        N_patch = Tp * Hp * Wp
        patch_tokens = normed[:, 1:] if normed.shape[1] == N_patch + 1 else normed
        # patch_tokens: [B, T'*H'*W', D]

        # Reshape to [B*T', H'*W', D] then attention pool → [B*T', D] → [B, T', D]
        tokens_per_t = patch_tokens.view(B * Tp, Hp * Wp, self.feat_dim)
        pooled_spatial = self.spatial_pool(tokens_per_t)  # [B*T', D]
        return pooled_spatial.view(B, Tp, self.feat_dim)  # [B, T', D]


class EarlyFusionMViTAttnPool(nn.Module):
    """
    MViT-v2-S early fusion backbone with attention pooling.

    Original: self._pooled_output.mean(dim=1)  → [B, D]
    This:     AttentionPool1D over all tokens  → [B, D]

    The pool attends over all N tokens (including [CLS] if present),
    letting the model focus on the most foul-relevant tokens rather
    than averaging everything including background.
    """

    def __init__(self, num_views: int = 5, T_per_view: int = 16):
        super().__init__()
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

        self.num_views = num_views
        self.T_fused = num_views * T_per_view
        self.feat_dim = 768
        self._pooled_output = None

        base = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)

        def _norm_hook(module, input, output):
            self._pooled_output = output

        base.norm.register_forward_hook(_norm_hook)
        self._base = base

        # Attention pool over tokens (replaces .mean(dim=1))
        self.token_pool = AttentionPool1D(feat_dim=768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, C, T, H, W = x.shape
        if T != 16:
            x = F.interpolate(x, size=(16, H, W), mode="trilinear", align_corners=False)

        self._base(x)
        tokens = self._pooled_output  # [B, N, 768]

        # Attention pool over all tokens → [B, 768]
        return self.token_pool(tokens)


class VideoMAEv2BackboneAttnPool(nn.Module):
    """
    VideoMAEv2 backbone with attention pooling replacing .mean(dim=1).

    Drop-in replacement for VideoMAEv2Backbone.
    Only change: last line uses self.token_pool instead of direct mean.
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
        self.pretrained_frames = 16
        self.fc = nn.Sequential()

        # Attention pool over sequence tokens (replaces .mean(dim=1))
        self.token_pool = AttentionPool1D(feat_dim=hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, C, T, H, W = x.shape
        assert C == 3
        if T != self.pretrained_frames:
            x = F.interpolate(
                x,
                size=(self.pretrained_frames, H, W),
                mode="trilinear",
                align_corners=False,
            )
        tokens = self.backbone(x)  # [B, N, D] — sequence of patch tokens

        return self.token_pool(tokens)  # [B, D]
