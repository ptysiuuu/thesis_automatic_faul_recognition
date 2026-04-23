import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBuilder(nn.Module):
    """
    Precomputes per-topology adjacency matrices and edge feature tensors for V=1..max_views.

    View convention
    ---------------
      View 0  = main live camera (privileged)
      Views 1+ = replay cameras

    Topologies
    ----------
      structured      All-pairs hub: replays ↔ live feed (bidirectional).
                      Replays also form a bidirectional chain (1↔2, 2↔3, …).
                      Captures the asymmetric live-vs-replay role.

      fully_connected All nodes connect to all others (ablation baseline).
                      Shows the benefit of structured inductive bias.

      replay_only     Replays → live feed only (no live→replay, no replay-replay).
                      Tests whether unidirectional aggregation is enough.

    All topologies add self-loops.

    Edge features  [V, V, 2]
    -------------------------
      dim 0  : 1 if self-loop  (i == j)
      dim 1  : 1 if the edge crosses view types  (live ↔ replay)
    """

    TOPOLOGIES = ('structured', 'fully_connected', 'replay_only')
    EDGE_FEAT_DIM = 2

    def __init__(self, max_views: int = 5, topology: str = 'structured'):
        super().__init__()
        if topology not in self.TOPOLOGIES:
            raise ValueError(f"topology must be one of {self.TOPOLOGIES}, got '{topology}'")
        self.topology = topology
        self.max_views = max_views

        for v in range(1, max_views + 1):
            adj, edge_attr = self._build(v)
            self.register_buffer(f'adj_{v}', adj)
            self.register_buffer(f'edge_attr_{v}', edge_attr)

    def _build(self, V: int):
        adj = torch.zeros(V, V)
        edge_attr = torch.zeros(V, V, self.EDGE_FEAT_DIM)

        def add_edge(src: int, dst: int):
            adj[src, dst] = 1.0
            edge_attr[src, dst, 0] = float(src == dst)               # self-loop flag
            edge_attr[src, dst, 1] = float((src == 0) != (dst == 0)) # cross-type flag

        # Self-loops always present
        for i in range(V):
            add_edge(i, i)

        if self.topology == 'fully_connected':
            for i in range(V):
                for j in range(V):
                    if i != j:
                        add_edge(i, j)

        elif self.topology == 'structured':
            for i in range(1, V):
                add_edge(i, 0)      # replay → live
                add_edge(0, i)      # live → replay
            for i in range(1, V - 1):
                add_edge(i, i + 1)  # adjacent replay → next
                add_edge(i + 1, i)  # next → adjacent replay

        elif self.topology == 'replay_only':
            for i in range(1, V):
                add_edge(i, 0)      # replay → live only

        return adj, edge_attr

    def get(self, V: int):
        return getattr(self, f'adj_{V}'), getattr(self, f'edge_attr_{V}')


class GATLayer(nn.Module):
    """
    Dense Graph Attention layer (Veličković et al., 2018) — pure PyTorch, no PyG.

    Uses a full [V_src, V_dst] attention matrix, efficient for V ≤ 5.

    For each edge i → j the attention coefficient is computed as:
        e_ij = LeakyReLU( a^T [W·h_i ‖ W·h_j] + edge_proj(f_ij) )
    then softmaxed over all incoming neighbours i for each target j.
    The output is:
        h_j' = ELU( Σ_i α_ij · W·h_i )  with a residual after the caller's LayerNorm.

    Parameters
    ----------
    feat_dim      : D — must be divisible by num_heads
    num_heads     : number of attention heads (outputs are concatenated then projected back to D)
    dropout       : attention weight dropout
    edge_feat_dim : dimensionality of the per-edge feature vector (matches GraphBuilder.EDGE_FEAT_DIM)

    Shapes
    ------
    x         : [B, V, D]
    adj       : [V, V]       — 1 where an edge exists, 0 otherwise
    edge_attr : [V, V, E]    — edge feature vectors (zeros for absent edges)
    mask      : [B, V]       — True for zero-padded (absent) views

    Returns
    -------
    h_new : [B, V, D]        — padded views are zero-filled
    """

    def __init__(self, feat_dim: int, num_heads: int = 4, dropout: float = 0.1,
                 edge_feat_dim: int = 2):
        super().__init__()
        if feat_dim % num_heads != 0:
            raise ValueError(f"feat_dim ({feat_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim  = feat_dim // num_heads
        self.feat_dim  = feat_dim

        # Shared linear projection applied to all nodes
        self.W        = nn.Linear(feat_dim, feat_dim, bias=False)
        # Per-head attention scalar from concatenated [src ‖ dst] head features
        self.attn     = nn.Linear(2 * self.head_dim, 1, bias=False)
        # Edge feature projection: E → num_heads (one additive bias per head per edge)
        self.edge_proj = nn.Linear(edge_feat_dim, num_heads, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.act     = nn.ELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                edge_attr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, V, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        Wx = self.W(x).view(B, V, H, Dh)                           # [B, V, H, Dh]

        # Expand to all (src, dst) pairs
        h_src = Wx.unsqueeze(2).expand(-1, -1, V, -1, -1)          # [B, Vsrc, Vdst, H, Dh]
        h_dst = Wx.unsqueeze(1).expand(-1, V, -1, -1, -1)          # [B, Vsrc, Vdst, H, Dh]

        # Raw attention scores: e[b, src, dst, head]
        cat = torch.cat([h_src, h_dst], dim=-1)                     # [B, V, V, H, 2*Dh]
        e   = self.attn(cat).squeeze(-1)                            # [B, V, V, H]

        # Add edge-feature bias (edge_proj of zeros = 0 for absent edges; bias=False)
        ef = self.edge_proj(edge_attr)                              # [V, V, H]
        e  = e + ef.unsqueeze(0)

        e = F.leaky_relu(e, negative_slope=0.2)

        # Mask absent edges → -inf so softmax ignores them
        non_edge = ~adj.bool().unsqueeze(0).unsqueeze(-1)           # [1, V, V, 1]
        e = e.masked_fill(non_edge, float('-inf'))

        # Mask padded nodes (both endpoints of every edge touching a padded view)
        pad_src = mask.unsqueeze(2).unsqueeze(-1)                   # [B, V, 1, 1]
        pad_dst = mask.unsqueeze(1).unsqueeze(-1)                   # [B, 1, V, 1]
        e = e.masked_fill(pad_src | pad_dst, float('-inf'))

        # Normalise: softmax over incoming sources for each target j
        alpha = torch.softmax(e, dim=1)                             # [B, Vsrc, Vdst, H]
        alpha = torch.nan_to_num(alpha, nan=0.0)                    # guard all-padded targets
        alpha = self.dropout(alpha)

        # Aggregate: h_new[j] = Σ_i  α[i,j] · W·h_i
        h_new = (alpha.unsqueeze(-1) * h_src).sum(dim=1)           # [B, Vdst, H, Dh]
        h_new = self.act(h_new.reshape(B, V, D))                   # [B, V, D]

        # Zero padded node outputs so they don't pollute residuals
        h_new = h_new.masked_fill(mask.unsqueeze(-1), 0.0)
        return h_new
