"""
cva_aggregate.py
================
Cross-View Attention (CVA) aggregator for multi-view foul recognition.

Inspired by MTV (Multiview Transformers for Video Recognition, CVPR 2022)
but adapted for:
  - PyTorch (not JAX/Flax)
  - Post-backbone aggregation (not within-backbone)
  - Variable view counts with padding masks
  - TAdaFormer [B, V, T', D] token input

Key difference from TransformerAggregate:
  TransformerAggregate: concatenates all view tokens → single transformer
  CVAAggregate: alternates within-view self-attention and cross-view attention
                maintaining per-view identity throughout, then fuses

Key difference from MTV:
  MTV CVA: sequential neighboring-view attention (view[i] attends to view[i+1])
  This CVA: each view attends to ALL other views (fully connected cross-view)
            This is more appropriate since our views don't have an ordering
            like MTV's slow/fast temporal views — they are camera angles.

Zero-initialization of cross-attention output projection (from MTV):
  Ensures the aggregator starts as pure within-view self-attention and
  gradually learns to incorporate cross-view information. More stable than
  random initialization.

Usage in MVAggregate (mvaggregate.py):
    elif agr_type == "cva":
        self.aggregation_model = CVAAggregate(
            model=model,
            feat_dim=feat_dim,
            lifting_net=lifting_net,
        )

And in MVNetwork (model.py) + checkArguments (main.py):
    Add "cva" to the list of valid pooling types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import batch_tensor, unbatch_tensor


class CrossViewAttentionLayer(nn.Module):
    """
    One layer of cross-view attention.

    For each view v, attends to tokens from all OTHER views at the same
    temporal positions. This is the core mechanism from MTV adapted to
    fully-connected (all-pairs) view topology instead of neighboring views.

    Architecture per layer (following MTV's ordering):
        1. LayerNorm → Cross-attention (view v queries all other views) → residual
        2. LayerNorm → Self-attention (within view v) → residual
        3. LayerNorm → MLP → residual

    The cross-attention output projection is zero-initialized so the layer
    starts as identity and learns cross-view communication gradually.

    Parameters
    ----------
    feat_dim  : D
    num_heads : attention heads (both self and cross attention)
    ffn_dim   : feedforward dimension in MLP block
    dropout   : dropout rate
    """

    def __init__(
        self,
        feat_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads

        # ── Cross-view attention (view v queries all other views) ────────────
        self.cross_norm = nn.LayerNorm(feat_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Zero-initialize output projection (MTV design choice for stability)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

        # ── Within-view self-attention ────────────────────────────────────────
        self.self_norm = nn.LayerNorm(feat_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ── MLP ───────────────────────────────────────────────────────────────
        self.mlp_norm = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, feat_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,      # [B, V, T', D]
        view_mask: torch.Tensor,   # [B, V] True = padded view
    ) -> torch.Tensor:
        """
        tokens    : [B, V, T', D]
        view_mask : [B, V] bool, True = padded (absent) view

        Returns   : [B, V, T', D] updated tokens
        """
        B, V, T, D = tokens.shape
        output = tokens.clone()

        # ── Step 1: Cross-view attention ──────────────────────────────────────
        # Each view v attends to tokens from all OTHER views
        # We process all views in parallel by masking self-view attention
        normed = self.cross_norm(tokens)  # [B, V, T', D]

        cross_out = torch.zeros_like(tokens)
        for v in range(V):
            if view_mask[:, v].all():
                # All batch items have this view padded — skip
                continue

            # Query: tokens of view v → [B, T', D]
            query = normed[:, v, :, :]  # [B, T', D]

            # Key/Value: tokens of all OTHER views → [B, (V-1)*T', D]
            other_views = [normed[:, u, :, :] for u in range(V) if u != v]
            if not other_views:
                continue
            kv = torch.cat(other_views, dim=1)  # [B, (V-1)*T', D]

            # Key padding mask: mark positions from padded views as True
            # other_view_masks: [B, (V-1)*T'] True = padded
            other_masks = [
                view_mask[:, u].unsqueeze(1).expand(-1, T)
                for u in range(V) if u != v
            ]
            kv_mask = torch.cat(other_masks, dim=1)  # [B, (V-1)*T']

            # If ALL key positions are padded for some batch items,
            # MultiheadAttention will produce NaN — handle gracefully
            all_kv_padded = kv_mask.all(dim=1)  # [B]

            attn_out, _ = self.cross_attn(
                query=query,
                key=kv,
                value=kv,
                key_padding_mask=kv_mask,
            )  # [B, T', D]

            # Zero out results where all keys were padded (NaN guard)
            attn_out = attn_out.masked_fill(
                all_kv_padded.unsqueeze(1).unsqueeze(2), 0.0
            )

            # Also zero out padded query views
            attn_out = attn_out.masked_fill(
                view_mask[:, v].unsqueeze(1).unsqueeze(2), 0.0
            )

            cross_out[:, v, :, :] = attn_out

        # Residual connection
        output = output + self.dropout(cross_out)

        # ── Step 2: Within-view self-attention ────────────────────────────────
        # Each view's tokens attend to each other (temporal self-attention)
        # Process all views together by reshaping to [B*V, T', D]
        BV = B * V
        x_flat = self.self_norm(output).reshape(BV, T, D)

        # Token padding mask: [B*V, T'] — all True for padded views
        # For non-padded views, all T' tokens are valid (no token-level padding)
        token_mask = view_mask.reshape(BV).unsqueeze(1).expand(-1, T)
        # MultiheadAttention key_padding_mask: True = ignore
        # We need per-sequence mask, not per-token, so use batch dimension
        # A fully-masked sequence (padded view) will produce NaN → guard below
        self_out, _ = self.self_attn(
            query=x_flat,
            key=x_flat,
            value=x_flat,
        )  # [B*V, T', D]

        # Zero padded views
        padded_flat = view_mask.reshape(BV)  # [B*V]
        self_out = self_out.masked_fill(
            padded_flat.unsqueeze(1).unsqueeze(2), 0.0
        )
        self_out = torch.nan_to_num(self_out, nan=0.0)

        output = output + self.dropout(self_out.reshape(B, V, T, D))

        # ── Step 3: MLP ───────────────────────────────────────────────────────
        mlp_in = self.mlp_norm(output).reshape(BV, T, D)
        mlp_out = self.mlp(mlp_in).reshape(B, V, T, D)
        mlp_out = mlp_out.masked_fill(
            view_mask.unsqueeze(2).unsqueeze(3), 0.0
        )
        output = output + mlp_out

        return output


class CVAAggregate(nn.Module):
    """
    Cross-View Attention aggregator.

    Replaces TransformerAggregate with a design that maintains per-view
    token sequences throughout the network and explicitly models cross-view
    relationships via alternating self- and cross-attention.

    Architecture:
        Backbone → [B, V, T', D]
        ↓
        View + temporal positional embeddings
        ↓
        Quality gate (per-view scalar)
        ↓
        N × CrossViewAttentionLayer
          (cross-view attn → within-view self-attn → MLP)
        ↓
        Temporal mean-pool → [B, V, D]
        ↓
        Cross-view fusion: [CLS] token attends to all view vectors
        ↓
        [B, D] representation

    The key advantage over TransformerAggregate: cross-view communication
    happens repeatedly throughout the stack (N layers), not just at the
    final pooling step. This allows the model to learn that the elbow
    contact visible in view 2 affects how it interprets the live camera
    in view 0.

    Parameters
    ----------
    model          : backbone (TAdaFormer or other)
    feat_dim       : D (768 for TAdaFormer-B/16)
    num_layers     : number of CVA layers (default 2)
    num_heads      : attention heads
    ffn_dim        : feedforward dimension
    lifting_net    : optional projection after backbone
    T_max          : maximum temporal tokens (8 for TAdaFormer)
    dropout        : dropout rate
    cascade_severity: if True, action logits are concatenated before severity
                      head (handled in MVAggregate, not here)
    """

    def __init__(
        self,
        model: nn.Module,
        feat_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        lifting_net: nn.Module = nn.Sequential(),
        T_max: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feat_dim = feat_dim
        self.T_max = T_max

        # ── Positional embeddings ────────────────────────────────────────────
        self.view_embeds = nn.Parameter(torch.zeros(1, 5, 1, feat_dim))
        self.temporal_embeds = nn.Parameter(torch.zeros(1, 1, T_max, feat_dim))
        nn.init.trunc_normal_(self.view_embeds, std=0.02)
        nn.init.trunc_normal_(self.temporal_embeds, std=0.02)

        # ── Quality gate (per-view scalar from mean-pooled features) ─────────
        self.quality_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, 1),
            nn.Sigmoid(),
        )

        # ── CVA layers ────────────────────────────────────────────────────────
        self.cva_layers = nn.ModuleList([
            CrossViewAttentionLayer(
                feat_dim=feat_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Final cross-view fusion: [CLS] attends to all view vectors ────────
        # After temporal mean-pool we have [B, V, D] — one vector per view
        # A learned [CLS] token aggregates across views
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(feat_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, mvimages: torch.Tensor):
        """
        mvimages : [B, V, C, T, H, W]

        Returns
        -------
        pooled     : [B, feat_dim]
        importance : [B, V] view importance weights (from fusion attention)
        """
        B, V, *_ = mvimages.shape

        # ── 1. Backbone feature extraction ───────────────────────────────────
        raw = unbatch_tensor(
            self.model(batch_tensor(mvimages, dim=1, squeeze=True)),
            B,
            dim=1,
            unsqueeze=True,
        )  # [B, V, D] or [B, V, T', D]

        # Ensure [B, V, T', D] shape
        if raw.dim() == 3:
            # Classic backbone returning [B, V, D] — add T'=1 dimension
            raw = raw.unsqueeze(2)  # [B, V, 1, D]
        T = raw.shape[2]

        raw = self.lifting_net(raw)  # [B, V, T', D] (identity if no lifting)

        # ── 2. View padding mask ──────────────────────────────────────────────
        view_mask = (
            mvimages.abs().sum(dim=tuple(range(2, mvimages.dim()))) == 0
        )  # [B, V] True = padded

        # ── 3. Quality gate ───────────────────────────────────────────────────
        quality = self.quality_gate(raw.mean(dim=2))  # [B, V, 1]
        quality = quality.masked_fill(view_mask.unsqueeze(-1), 0.0)
        raw = raw * (0.5 + quality.unsqueeze(2))  # [B, V, T', D]

        # ── 4. Positional embeddings ──────────────────────────────────────────
        raw = raw + self.view_embeds[:, :V, :, :]       # view position
        raw = raw + self.temporal_embeds[:, :, :T, :]   # temporal position

        # ── 5. CVA layers ─────────────────────────────────────────────────────
        x = raw  # [B, V, T', D]
        for layer in self.cva_layers:
            x = layer(x, view_mask)  # [B, V, T', D]

        # ── 6. Temporal mean-pool → [B, V, D] ─────────────────────────────────
        # Mask padded views before pooling
        x_masked = x.masked_fill(view_mask.unsqueeze(2).unsqueeze(3), 0.0)
        view_features = x_masked.mean(dim=2)  # [B, V, D]

        # ── 7. Cross-view fusion: [CLS] attends to view vectors ───────────────
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

        pooled, importance = self.fusion_attn(
            query=cls,
            key=view_features,
            value=view_features,
            key_padding_mask=view_mask,
        )  # pooled: [B, 1, D], importance: [B, 1, V]

        pooled = self.fusion_norm(pooled.squeeze(1))  # [B, D]
        importance = importance.squeeze(1)            # [B, V]

        return pooled, importance
