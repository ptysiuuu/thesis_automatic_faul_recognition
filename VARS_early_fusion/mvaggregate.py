from utils import batch_tensor, unbatch_tensor
import torch
from torch import nn
from graph import GraphBuilder, GATLayer


class WeightedAggregate(nn.Module):
    def __init__(self, model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feature_dim = feat_dim

        r1, r2 = -1, 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)
        self.normReLu = nn.Sequential(nn.LayerNorm(feat_dim), nn.ReLU())
        self.relu = nn.ReLU()

    def forward(self, mvimages):
        B, V, *_ = mvimages.shape
        aux = self.lifting_net(
            unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
        )  # [B, V, feat_dim] or [B, V, T', feat_dim]
        if aux.dim() == 4:
            aux = aux.mean(dim=2)   # temporal mean-pool → [B, V, feat_dim]

        aux = torch.matmul(aux, self.attention_weights)
        aux_t = aux.permute(0, 2, 1)
        prod = torch.bmm(aux, aux_t)
        relu_res = self.relu(prod)

        aux_sum = torch.sum(torch.reshape(relu_res, (B, V * V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V * V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T.reshape(B, V, V).sum(1)

        output = torch.sum(torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1)), 1)
        return output.squeeze(), final_attention_weights


class TransformerAggregate(nn.Module):
    """
    TransformerAggregate with view-quality gating and temporal token support.

    Handles both classic backbones (output [B*V, D] → [B, V, D]) and
    temporal-token backbones like MViT-v2-S (output [B*V, T', D] → [B, V, T', D]).
    In the temporal case each (view, time-step) pair becomes one sequence token,
    giving V*T' = 5*8 = 40 tokens before the [CLS] token.
    """

    def __init__(self, model, feat_dim, num_layers=1, num_heads=4,
                 lifting_net=nn.Sequential(), T_max=8):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feat_dim = feat_dim

        # [CLS] token — aggregates information from all views / time-steps
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))

        # Positional embeddings: one per view (max 5) and one per time-step
        self.view_embeds     = nn.Parameter(torch.zeros(1, 5,     feat_dim))
        self.temporal_embeds = nn.Parameter(torch.zeros(1, T_max, feat_dim))

        # View-quality gating: scalar per view, learned from mean-pooled features
        self.quality_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, 1),
            nn.Sigmoid(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=num_heads,
            batch_first=True, dropout=0.1, dim_feedforward=feat_dim,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        nn.init.trunc_normal_(self.cls_token,      std=0.02)
        nn.init.trunc_normal_(self.view_embeds,     std=0.02)
        nn.init.trunc_normal_(self.temporal_embeds, std=0.02)

    def forward(self, mvimages):
        B, V, *_ = mvimages.shape

        # --- backbone feature extraction ---
        raw = unbatch_tensor(
            self.model(batch_tensor(mvimages, dim=1, squeeze=True)),
            B, dim=1, unsqueeze=True,
        )
        # raw: [B, V, D] for classic backbones  OR  [B, V, T', D] for MViT-v2-S

        if raw.dim() == 3:                    # classic path: add dummy time axis
            raw = self.lifting_net(raw)       # [B, V, D]
            raw = raw.unsqueeze(2)            # [B, V, 1, D]
        # raw is now always [B, V, T', D]
        T = raw.shape[2]

        # --- view padding mask ---
        view_mask = (mvimages.abs().sum(dim=(2, 3, 4, 5)) == 0)  # [B, V] True = padded

        # --- per-view quality gate (from temporal mean) ---
        quality = self.quality_gate(raw.mean(2))                  # [B, V, 1]
        quality = quality.masked_fill(view_mask.unsqueeze(-1), 0.0)
        raw = raw * (0.5 + quality.unsqueeze(2))                  # [B, V, T', D]

        # --- positional embeddings ---
        raw = raw + self.view_embeds[:, :V, :].unsqueeze(2)       # broadcast over T'
        raw = raw + self.temporal_embeds[:, :T, :].unsqueeze(1)   # broadcast over V

        # --- flatten (view, time) → sequence ---
        tokens = raw.flatten(1, 2)                                # [B, V*T', D]

        # expand view padding mask to cover all T' tokens per view
        pad_token_mask = (
            view_mask.unsqueeze(2).expand(-1, -1, T).flatten(1, 2)  # [B, V*T']
        )

        # --- prepend [CLS] ---
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)                # [B, V*T'+1, D]
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=mvimages.device)
        padding_mask = torch.cat([cls_mask, pad_token_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return x[:, 0], None


class CrossAttentionAggregate(nn.Module):
    def __init__(self, model, feat_dim, num_heads=4, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, mvimages):
        B, *_ = mvimages.shape
        aux = self.lifting_net(
            unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
        )  # [B, V, feat_dim] or [B, V, T', feat_dim]
        if aux.dim() == 4:
            aux = aux.mean(dim=2)   # temporal mean-pool → [B, V, feat_dim]

        main_cam = aux[:, 0:1, :]
        replays = aux[:, 1:, :]
        replay_mask = (mvimages[:, 1:, :].abs().sum(dim=(2, 3, 4, 5)) == 0)

        if replay_mask.all():
            return self.norm(main_cam.squeeze(1)), None

        attn_out, attn_weights = self.cross_attn(
            query=main_cam, key=replays, value=replays, key_padding_mask=replay_mask
        )
        output = self.norm(main_cam + attn_out)
        return output.squeeze(1), attn_weights


class ViewMaxAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, *_ = mvimages.shape
        aux = self.lifting_net(
            unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
        )  # [B, V, feat_dim] or [B, V, T', feat_dim]
        if aux.dim() == 4:
            aux = aux.mean(dim=2)   # temporal mean-pool → [B, V, feat_dim]
        return torch.max(aux, dim=1)[0].squeeze(), aux


class GATAggregate(nn.Module):
    """
    Graph Attention Network aggregation over multi-view features.

    Pipeline
    --------
      1. Per-view backbone features  [B, V, D]  (shared weights via batch_tensor).
      2. View-quality gating: learned scalar per view suppresses low-quality views.
      3. Two GAT layers (residual + LayerNorm) with the chosen graph topology.
      4. Importance-weighted readout: learned scalar per view, softmaxed, then
         weighted sum → [B, D].

    The graph topology is fixed (structured / fully_connected / replay_only) and
    precomputed by GraphBuilder for each V so there is no per-forward overhead.
    """

    def __init__(self, model, feat_dim: int, num_heads: int = 4, num_layers: int = 2,
                 lifting_net=nn.Sequential(), topology: str = 'structured'):
        super().__init__()
        self.model        = model
        self.lifting_net  = lifting_net
        self.feat_dim     = feat_dim

        self.graph_builder = GraphBuilder(max_views=5, topology=topology)
        self.gat_layers    = nn.ModuleList([
            GATLayer(feat_dim, num_heads=num_heads, dropout=0.1,
                     edge_feat_dim=GraphBuilder.EDGE_FEAT_DIM)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(num_layers)])

        self.quality_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Scalar importance score per node; softmax → weighted pooling
        self.readout = nn.Linear(feat_dim, 1)

    def forward(self, mvimages: torch.Tensor):
        B, V, *_ = mvimages.shape

        # --- backbone: shared weights across views ---
        raw = unbatch_tensor(
            self.model(batch_tensor(mvimages, dim=1, squeeze=True)),
            B, dim=1, unsqueeze=True,
        )
        # raw: [B, V, T', D] for MViT-v2-S  OR  [B, V, D] for classic backbones
        if raw.dim() == 4:
            raw = raw.mean(dim=2)       # temporal mean → [B, V, D]
        raw = self.lifting_net(raw)     # identity by default

        # --- padding mask: True = view is all-zeros (absent) ---
        view_mask = (mvimages.abs().sum(dim=tuple(range(2, mvimages.dim()))) == 0)  # [B, V]

        # --- quality gate ---
        quality = self.quality_gate(raw)                            # [B, V, 1]
        quality = quality.masked_fill(view_mask.unsqueeze(-1), 0.0)
        x = raw * (0.5 + quality)                                   # [B, V, D]

        # --- precomputed graph for this V ---
        adj, edge_attr = self.graph_builder.get(V)                  # [V,V], [V,V,2]

        # --- GAT layers with pre-norm residual ---
        for gat, norm in zip(self.gat_layers, self.norms):
            x = norm(x + gat(x, adj, edge_attr, view_mask))

        # --- importance-weighted readout ---
        scores = self.readout(x).squeeze(-1)                        # [B, V]
        scores = scores.masked_fill(view_mask, float('-inf'))
        importance = torch.softmax(scores, dim=-1)
        importance = torch.nan_to_num(importance, nan=0.0)

        pooled = (x * importance.unsqueeze(-1)).sum(dim=1)          # [B, D]
        return pooled, importance


class MVAggregate(nn.Module):
    """
    Multi-view aggregation module with:
      - ordinal severity head  (3 logits → 4 ordinal classes)
      - action head            (8-class CE)
      - 4 auxiliary heads      (contact, bodypart, try_to_play, handball) — BCE
    """

    def __init__(self, model, agr_type="transformer", feat_dim=400,
                 lifting_net=nn.Sequential(), graph_topology="structured",
                 cascade_severity=False):
        super().__init__()
        self.agr_type = agr_type
        self.cascade_severity = cascade_severity

        # Shared intermediate projection
        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

        # --- primary heads ---
        # Ordinal severity: when cascade_severity=True, action logits (8-dim) are
        # concatenated with the aggregator output before this head.
        sev_in = feat_dim + 8 if cascade_severity else feat_dim
        self.fc_ordinal_severity = nn.Sequential(
            nn.LayerNorm(sev_in),
            nn.Dropout(p=0.3),
            nn.Linear(sev_in, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim // 2, 3),   # 3 cumulative logits
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, 8),
        )

        # --- auxiliary heads (single logit each, trained with BCEWithLogitsLoss) ---
        self.fc_contact = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1)
        )
        self.fc_bodypart = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1)
        )
        self.fc_try_to_play = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1)
        )
        self.fc_handball = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1)
        )

        # --- aggregator ---
        if agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif agr_type == "transformer":
            self.aggregation_model = TransformerAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)
        elif agr_type == "crossattn":
            self.aggregation_model = CrossAttentionAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)
        elif agr_type == "gat":
            self.aggregation_model = GATAggregate(model=model, feat_dim=feat_dim,
                                                  lifting_net=lifting_net, topology=graph_topology)
        else:
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)

    def forward(self, mvimages):
        pooled_view, attention = self.aggregation_model(mvimages)  # [B, feat_dim]
        inter = self.inter(pooled_view)                            # [B, feat_dim]

        pred_action = self.fc_action(inter)                        # [B, 8]
        if self.cascade_severity:
            sev_in = torch.cat([inter, pred_action], dim=-1)       # [B, feat_dim+8]
            pred_ordinal_severity = self.fc_ordinal_severity(sev_in)
        else:
            pred_ordinal_severity = self.fc_ordinal_severity(inter)
        pred_contact = self.fc_contact(inter).squeeze(-1)          # [B]
        pred_bodypart = self.fc_bodypart(inter).squeeze(-1)        # [B]
        pred_try_to_play = self.fc_try_to_play(inter).squeeze(-1)  # [B]
        pred_handball = self.fc_handball(inter).squeeze(-1)        # [B]

        return (pred_ordinal_severity, pred_action,
                pred_contact, pred_bodypart, pred_try_to_play, pred_handball,
                attention)
