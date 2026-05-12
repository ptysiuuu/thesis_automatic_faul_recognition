import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import batch_tensor, unbatch_tensor
from graph import TokenGraphBuilder, GATLayer


class SetNorm(nn.Module):
    """
    Permutation-invariant normalization for sets.
    Normalizes across both the set (V or V*T) and feature dimensions jointly.
    """

    def __init__(self, feat_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(feat_dim))
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.scale + self.bias


class TemporalLocalizer(nn.Module):
    """
    Learns attention weights over the temporal dimension for each view.
    Returns view-level features and temporal weights [B, V, T].
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.GELU(),
            nn.Linear(feat_dim // 4, 1),
        )
        nn.init.zeros_(self.scorer[0].weight)
        nn.init.zeros_(self.scorer[2].weight)
        nn.init.zeros_(self.scorer[0].bias)
        nn.init.zeros_(self.scorer[2].bias)

    def forward(self, x: torch.Tensor, view_mask: torch.Tensor):
        # x: [B, V, T, D]
        scores = self.scorer(x).squeeze(-1)
        scores = scores.masked_fill(view_mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weighted = (x * weights.unsqueeze(-1)).sum(dim=2)
        return weighted, weights


class MambaBlock(nn.Module):
    """
    Simplified Mamba-style SSM block with diagonal A and ZOH discretization.
    """

    def __init__(self, d_model: int, expand: int = 2, conv_kernel: int = 4):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = d_model * expand
        self.conv_kernel = conv_kernel

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            bias=True,
        )
        self.B_proj = nn.Linear(self.d_inner, self.d_inner)
        self.C_proj = nn.Linear(self.d_inner, self.d_inner)
        self.gate_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # Diagonal A with stable negative values.
        init_a = torch.linspace(1.0, 0.1, self.d_inner)
        self.log_A = nn.Parameter(torch.log(init_a))
        self.delta = nn.Parameter(torch.tensor(0.1))

    def _causal_conv(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]
        pad = self.conv_kernel - 1
        x = F.pad(x, (pad, 0))
        x = self.conv(x)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        proj = self.in_proj(x)
        x_in, gate_in = proj.chunk(2, dim=-1)

        x_conv = self._causal_conv(x_in)
        B_t = self.B_proj(x_conv)
        C_t = self.C_proj(x_conv)
        gate = torch.sigmoid(self.gate_proj(gate_in))

        delta = F.softplus(self.delta)
        A = -torch.exp(self.log_A)  # [D_inner]
        A_bar = torch.exp(A * delta)
        A_denom = torch.where(A.abs() < 1e-6, torch.full_like(A, -1e-6), A)
        B_bar = (A_bar - 1.0) / A_denom

        B, L, _ = x_conv.shape
        h = torch.zeros(B, self.d_inner, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = A_bar * h + B_bar * B_t[:, t, :]
            y = C_t[:, t, :] * h
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y * gate
        return self.out_proj(y)


class BidirectionalSSMBlock(nn.Module):
    """
    Four-pass bidirectional SSM over V x T grid.
    """

    def __init__(self, feat_dim: int, expand: int = 2, conv_kernel: int = 4):
        super().__init__()
        self.ssm = MambaBlock(feat_dim, expand=expand, conv_kernel=conv_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, V, T, D]
        B, V, T, D = x.shape

        # View-prioritized: [t0v0, t0v1, ...]
        seq_view = x.permute(0, 2, 1, 3).contiguous().view(B, T * V, D)
        out_view = self.ssm(seq_view)
        out_view_rev = self.ssm(torch.flip(seq_view, dims=[1]))
        out_view_rev = torch.flip(out_view_rev, dims=[1])

        # Temporal-prioritized: [v0t0, v0t1, ...]
        seq_temp = x.contiguous().view(B, V * T, D)
        out_temp = self.ssm(seq_temp)
        out_temp_rev = self.ssm(torch.flip(seq_temp, dims=[1]))
        out_temp_rev = torch.flip(out_temp_rev, dims=[1])

        # Unflatten and align
        out_view = out_view.view(B, T, V, D).permute(0, 2, 1, 3)
        out_view_rev = out_view_rev.view(B, T, V, D).permute(0, 2, 1, 3)
        out_temp = out_temp.view(B, V, T, D)
        out_temp_rev = out_temp_rev.view(B, V, T, D)

        return x + (out_view + out_view_rev + out_temp + out_temp_rev)


class MVGMNBlock(nn.Module):
    """
    One MV-GMN block: bidirectional SSM + token-level GCN.
    """

    def __init__(
        self,
        feat_dim: int,
        num_heads: int = 4,
        knn_k: int = 3,
        topology: str = "structured",
        max_views: int = 5,
        T_max: int = 8,
    ):
        super().__init__()
        self.ssm_block = BidirectionalSSMBlock(feat_dim)
        self.knn_k = knn_k
        self.token_graph = TokenGraphBuilder(
            max_views=max_views, T_max=T_max, topology=topology
        )
        self.gcn = GATLayer(
            feat_dim=feat_dim, num_heads=num_heads, dropout=0.1, edge_feat_dim=4
        )

    def _build_knn(self, x: torch.Tensor, k: int) -> torch.Tensor:
        # x: [N, D]
        N = x.shape[0]
        if k <= 0 or N <= 1:
            return torch.zeros(N, N, device=x.device)
        x_norm = F.normalize(x, dim=-1, eps=1e-6)
        sim = x_norm @ x_norm.t()
        sim.fill_diagonal_(-1e9)
        k_eff = min(k, N - 1)
        topk = torch.topk(sim, k_eff, dim=-1).indices
        adj = torch.zeros(N, N, device=x.device)
        adj.scatter_(1, topk, 1.0)
        return adj

    def _merge_edge_attr(
        self, edge_attr_rule: torch.Tensor, adj_knn: torch.Tensor
    ) -> torch.Tensor:
        N = edge_attr_rule.shape[0]
        edge_attr = torch.zeros(N, N, 4, device=edge_attr_rule.device)
        edge_attr[:, :, :3] = edge_attr_rule
        edge_attr[:, :, 3] = adj_knn
        return edge_attr

    def forward(self, x: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, V, T, D]
        x = self.ssm_block(x)
        B, V, T, D = x.shape
        N = V * T

        x_flat = x.reshape(B, N, D)
        token_mask = view_mask.unsqueeze(2).expand(-1, -1, T).reshape(B, N)

        adj_rule, edge_attr_rule = self.token_graph.get(V, T)
        adj_rule = adj_rule.to(x.device)
        edge_attr_rule = edge_attr_rule.to(x.device)

        outs = []
        for b in range(B):
            xb = x_flat[b : b + 1]
            mask_b = token_mask[b : b + 1]

            adj_knn = self._build_knn(xb[0], self.knn_k)
            adj = (adj_rule.bool() | adj_knn.bool()).float()
            edge_attr = self._merge_edge_attr(edge_attr_rule, adj_knn)

            out_b = self.gcn(xb, adj, edge_attr, mask_b)
            outs.append(out_b)

        out = torch.cat(outs, dim=0).view(B, V, T, D)
        return x + out


class TokenWeightedPool(nn.Module):
    """
    Learned weighted pooling over V*T tokens.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, view_mask: torch.Tensor):
        # x: [B, V, T, D]
        B, V, T, D = x.shape
        tokens = x.reshape(B, V * T, D)
        scores = self.score(tokens).squeeze(-1)

        token_mask = view_mask.unsqueeze(2).expand(-1, -1, T).reshape(B, V * T)
        scores = scores.masked_fill(token_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = (weights.unsqueeze(-1) * tokens).sum(dim=1)
        return pooled, weights


class MVGMNAggregate(nn.Module):
    """
    MV-GMN aggregator with bidirectional SSM blocks and token-level GCN.

    Input : mvimages [B, V, C, T, H, W]
    Output: pooled [B, D], temporal_weights [B, V, T]
    """

    def __init__(
        self,
        model,
        feat_dim: int,
        lifting_net=nn.Sequential(),
        num_blocks: int = 2,
        num_heads: int = 4,
        knn_k: int = 3,
        topology: str = "structured",
        T_max: int = 8,
    ):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feat_dim = feat_dim
        self.set_norm = SetNorm(feat_dim)
        self.temporal_localizer = TemporalLocalizer(feat_dim)

        self.quality_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, 1),
            nn.Sigmoid(),
        )

        self.blocks = nn.ModuleList(
            [
                MVGMNBlock(
                    feat_dim=feat_dim,
                    num_heads=num_heads,
                    knn_k=knn_k,
                    topology=topology,
                    max_views=5,
                    T_max=T_max,
                )
                for _ in range(num_blocks)
            ]
        )

        self.pool = TokenWeightedPool(feat_dim)

    def forward(self, mvimages: torch.Tensor):
        B, V, *_ = mvimages.shape

        raw = unbatch_tensor(
            self.model(batch_tensor(mvimages, dim=1, squeeze=True)),
            B,
            dim=1,
            unsqueeze=True,
        )

        if raw.dim() == 3:
            raw = self.lifting_net(raw)
            raw = raw.unsqueeze(2)
        T = raw.shape[2]

        raw_flat = raw.flatten(1, 2)
        raw_flat = self.set_norm(raw_flat)
        raw = raw_flat.view(B, V, T, -1)

        view_mask = mvimages.abs().sum(dim=(2, 3, 4, 5)) == 0

        view_features, temporal_weights = self.temporal_localizer(raw, view_mask)
        quality = self.quality_gate(view_features)
        quality = quality.masked_fill(view_mask.unsqueeze(-1), 0.0)
        raw = raw * (0.5 + quality.unsqueeze(2))

        x = raw
        for block in self.blocks:
            x = block(x, view_mask)

        pooled, _ = self.pool(x, view_mask)
        return pooled, temporal_weights
