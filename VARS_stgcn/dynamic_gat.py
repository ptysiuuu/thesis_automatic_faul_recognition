import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import batch_tensor, unbatch_tensor


class DynamicEdgeBuilder(nn.Module):
    """
    Builds a soft adjacency matrix from cosine similarity between node features.

    Given node features [B, V, D], computes pairwise cosine similarity
    and keeps top-K edges per node (others set to 0).

    Returns
    -------
    soft_adj : [B, V, V]  values in [0, 1], soft edge weights
               diagonal (self-loops) always 1.0
    """

    def __init__(self, k: int = 2, temperature: float = 0.1):
        """
        k           : number of nearest neighbours to connect (excluding self)
        temperature : softmax temperature for edge weight sharpness
                      lower = sharper (more selective), higher = softer
        """
        super().__init__()
        self.k = k
        self.temperature = temperature

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x    : [B, V, D]  node features (mean-pooled over time if temporal)
        mask : [B, V] bool, True = padded view

        Returns soft_adj [B, V, V]
        """
        B, V, D = x.shape

        # Cosine similarity matrix [B, V, V]
        x_norm = F.normalize(x, dim=-1)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, V, V]

        # Zero out padded nodes (both as source and target)
        if mask is not None:
            sim = sim.masked_fill(mask.unsqueeze(2), -1e9)
            sim = sim.masked_fill(mask.unsqueeze(1), -1e9)

        # Keep only top-K neighbours per node (excluding self)
        # Set self-similarity to -inf so it doesn't count as a neighbour
        sim_no_self = sim.clone()
        diag_mask = torch.eye(V, dtype=torch.bool, device=x.device).unsqueeze(0)
        sim_no_self = sim_no_self.masked_fill(diag_mask, -1e9)

        # Find K-th largest similarity per row
        k = min(self.k, V - 1)
        if k > 0:
            topk_vals, _ = sim_no_self.topk(k, dim=-1)  # [B, V, K]
            threshold = topk_vals[:, :, -1:]  # [B, V, 1]  K-th value

            # Binary mask: only keep edges above threshold
            keep = sim_no_self >= threshold  # [B, V, V]

            # Soft weights: softmax over kept neighbours
            sim_masked = sim_no_self.masked_fill(~keep, -1e9)
            soft_adj = torch.softmax(sim_masked / self.temperature, dim=-1)
            soft_adj = torch.nan_to_num(soft_adj, nan=0.0)
        else:
            soft_adj = torch.zeros(B, V, V, device=x.device)

        # Add self-loops with weight 1.0
        soft_adj = soft_adj + diag_mask.float()

        return soft_adj  # [B, V, V]


# =============================================================================
# GAT layer that handles soft (weighted) adjacency
# =============================================================================


class SoftGATLayer(nn.Module):
    """
    GAT layer for soft/weighted adjacency matrices.

    Unlike the original GATLayer (which uses binary adj + learned attention),
    this layer uses the pre-computed soft edge weights directly as attention
    coefficients — no additional attention network needed.

    The soft_adj already encodes semantic similarity, so we just use it
    as attention weights over the neighbourhood.

    h_j' = ELU( Σ_i  soft_adj[i,j] · W·h_i )

    Parameters
    ----------
    feat_dim  : D
    num_heads : H (outputs concatenated then projected back to D)
    dropout   : applied to soft_adj weights
    """

    def __init__(self, feat_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if feat_dim % num_heads != 0:
            raise ValueError(
                f"feat_dim ({feat_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.feat_dim = feat_dim

        self.W = nn.Linear(feat_dim, feat_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        soft_adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x        : [B, V, D]
        soft_adj : [B, V, V]  soft attention weights (already normalised per row)
        mask     : [B, V]     True = padded
        """
        B, V, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        Wx = self.W(x).view(B, V, H, Dh)  # [B, V, H, Dh]

        # Expand for all (src, dst) pairs
        h_src = Wx.unsqueeze(2).expand(-1, -1, V, -1, -1)  # [B, Vsrc, Vdst, H, Dh]

        # Soft adjacency as per-head weights: [B, V, V] → [B, V, V, H]
        alpha = soft_adj.unsqueeze(-1).expand(-1, -1, -1, H)  # [B, Vsrc, Vdst, H]
        alpha = self.dropout(alpha)

        # Zero out padded sources
        pad_src = mask.unsqueeze(2).unsqueeze(-1)  # [B, V, 1, 1]
        alpha = alpha.masked_fill(pad_src, 0.0)

        # Aggregate: h_new[j] = Σ_i  alpha[i,j] · W·h_i
        h_new = (alpha.unsqueeze(-1) * h_src).sum(dim=1)  # [B, Vdst, H, Dh]
        h_new = self.act(h_new.reshape(B, V, D))  # [B, V, D]

        # Zero padded outputs
        h_new = h_new.masked_fill(mask.unsqueeze(-1), 0.0)
        return h_new


# =============================================================================
# Main aggregator
# =============================================================================


class DynamicGATAggregate(nn.Module):
    """
    GAT aggregator with dual edge types:
      - Structural edges (fixed live/replay topology, from GraphBuilder)
      - Dynamic KNN edges (per-sample cosine similarity between views)

    Both edge types run in parallel through separate GATLayer / SoftGATLayer
    instances. Their outputs are combined via a learned gate.

    This operates at the VIEW level (not token level) for clarity and
    efficiency. If you want token-level dynamic edges, see note at bottom.

    Parameters
    ----------
    model          : backbone
    feat_dim       : D
    num_heads      : attention heads for both GAT types
    num_layers     : number of stacked dual-GAT layers
    topology       : structural topology ('fully_connected', 'structured', etc.)
    knn_k          : number of dynamic nearest neighbours (default 2)
    knn_temperature: softmax temperature for dynamic edge weights
    lifting_net    : post-backbone projection
    """

    def __init__(
        self,
        model,
        feat_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        lifting_net: nn.Module = nn.Sequential(),
        topology: str = "structured",
        knn_k: int = 2,
        knn_temperature: float = 0.1,
        T_max: int = 8,
    ):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feat_dim = feat_dim
        self.T_max = T_max

        # --- Structural graph (fixed topology, from original code) ---
        from graph import GraphBuilder, GATLayer

        self.graph_builder = GraphBuilder(max_views=5, topology=topology)

        # --- Structural GAT layers (existing, unchanged) ---
        self.struct_gat = nn.ModuleList(
            [
                GATLayer(
                    feat_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                    edge_feat_dim=GraphBuilder.EDGE_FEAT_DIM,
                )
                for _ in range(num_layers)
            ]
        )

        # --- Dynamic edge builder ---
        self.dynamic_edge_builder = DynamicEdgeBuilder(
            k=knn_k, temperature=knn_temperature
        )

        # --- Dynamic GAT layers (soft adjacency) ---
        self.dynamic_gat = nn.ModuleList(
            [
                SoftGATLayer(feat_dim, num_heads=num_heads, dropout=0.1)
                for _ in range(num_layers)
            ]
        )

        # --- Per-layer fusion gate: combines struct and dynamic outputs ---
        # Gate input: [struct_out, dynamic_out] → 2 scalars → softmax
        self.layer_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feat_dim * 2, feat_dim // 4),
                    nn.ReLU(),
                    nn.Linear(feat_dim // 4, 2),
                    nn.Softmax(dim=-1),
                )
                for _ in range(num_layers)
            ]
        )

        # --- Layer norms ---
        self.norms = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(num_layers)])

        # --- Positional embeddings (same as original GATAggregate) ---
        self.view_embeds = nn.Parameter(torch.zeros(1, 5, feat_dim))
        self.temporal_embeds = nn.Parameter(torch.zeros(1, T_max, feat_dim))
        nn.init.trunc_normal_(self.view_embeds, std=0.02)
        nn.init.trunc_normal_(self.temporal_embeds, std=0.02)

        # --- Quality gate (same as original) ---
        self.quality_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, 1),
            nn.Sigmoid(),
        )

        # --- Readout: attention pooling over views → [B, D] ---
        self.readout_query = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.readout_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.readout_norm = nn.LayerNorm(feat_dim)
        nn.init.trunc_normal_(self.readout_query, std=0.02)

    # ------------------------------------------------------------------
    def forward(self, mvimages: torch.Tensor):
        """
        mvimages : [B, V, C, T, H, W]

        Returns
        -------
        pooled     : [B, feat_dim]
        importance : [B, V]  readout attention weights
        """
        B, V, *_ = mvimages.shape

        # --- 1. Backbone extraction ---
        raw = unbatch_tensor(
            self.model(batch_tensor(mvimages, dim=1, squeeze=True)),
            B,
            dim=1,
            unsqueeze=True,
        )

        if raw.dim() == 3:
            raw = raw.unsqueeze(2)  # [B, V, 1, D]
        T = raw.shape[2]

        raw = self.lifting_net(raw)

        # --- 2. View padding mask ---
        view_mask = (
            mvimages.abs().sum(dim=tuple(range(2, mvimages.dim()))) == 0
        )  # [B, V]

        # --- 3. Quality gate ---
        quality = self.quality_gate(raw.mean(dim=2))  # [B, V, 1]
        quality = quality.masked_fill(view_mask.unsqueeze(-1), 0.0)
        raw = raw * (0.5 + quality.unsqueeze(2))

        # --- 4. Positional embeddings ---
        raw = raw + self.view_embeds[:, :V, :].unsqueeze(2)
        raw = raw + self.temporal_embeds[:, :T, :].unsqueeze(1)

        # --- 5. Temporal mean-pool for view-level operations ---
        # Both GAT types operate at view level (not token level) for efficiency
        x = raw.mean(dim=2)  # [B, V, D]  temporal mean

        # --- 6. Pre-compute dynamic edges (once, from initial features) ---
        # Use features BEFORE any GAT processing so edges reflect backbone features
        dynamic_adj = self.dynamic_edge_builder(x, mask=view_mask)  # [B, V, V]

        # --- 7. Structural edges (fixed, from GraphBuilder) ---
        struct_adj, edge_attr = self.graph_builder.get(V)  # [V, V], [V, V, E]

        # --- 8. Dual GAT layers ---
        for struct_layer, dynamic_layer, gate, norm in zip(
            self.struct_gat, self.dynamic_gat, self.layer_gates, self.norms
        ):
            # Structural path
            h_struct = struct_layer(x, struct_adj, edge_attr, view_mask)  # [B, V, D]

            # Dynamic path
            h_dynamic = dynamic_layer(x, dynamic_adj, view_mask)  # [B, V, D]

            # Per-node fusion gate
            # Gate input: [B, V, 2D] → weights [B, V, 2]
            gate_input = torch.cat([h_struct, h_dynamic], dim=-1)
            gate_weights = gate(gate_input)  # [B, V, 2]
            h_fused = (
                gate_weights[:, :, 0:1] * h_struct + gate_weights[:, :, 1:2] * h_dynamic
            )  # [B, V, D]

            # Residual + norm
            x = norm(x + h_fused)

        # --- 9. Readout: PMA-style attention pool over views → [B, D] ---
        seed = self.readout_query.expand(B, -1, -1)  # [B, 1, D]
        pooled, importance = self.readout_attn(
            query=seed,
            key=x,
            value=x,
            key_padding_mask=view_mask,
        )
        pooled = self.readout_norm(pooled.squeeze(1))  # [B, D]

        return pooled, importance.squeeze(1)  # [B, D], [B, V]
