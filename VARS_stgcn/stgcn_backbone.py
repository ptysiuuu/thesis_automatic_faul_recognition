import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_BASE_JOINTS = {
    "coco17": 17,
    "openpose18": 18,
}

_DEFAULT_BALL_KEYS = {
    "coco17": [9, 10, 15, 16],
    "openpose18": [4, 7, 11, 14],
}


def _build_skeleton_edges(layout: str, num_joints: int):
    if layout == "coco17":
        edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (0, 5),
            (0, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 6),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]
    elif layout == "openpose18":
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 5),
            (5, 6),
            (6, 7),
            (1, 8),
            (8, 9),
            (9, 10),
            (1, 11),
            (11, 12),
            (12, 13),
            (0, 14),
            (14, 16),
            (0, 15),
            (15, 17),
        ]
    else:
        edges = [(i, i + 1) for i in range(num_joints - 1)]

    return [(i, j) for i, j in edges if i < num_joints and j < num_joints]


def _normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(dim=1).clamp(min=1.0)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    d_mat = torch.diag(deg_inv_sqrt)
    return d_mat @ adj @ d_mat


def _normalize_adjacency_dynamic(adj: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(dim=-1).clamp(min=1.0)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    return adj * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)


def _infer_ball_index(layout: str, num_joints: int, ball_joint_index):
    if ball_joint_index is not None:
        return ball_joint_index
    base = _BASE_JOINTS.get(layout)
    if base is None:
        return None
    if num_joints == base + 1:
        return num_joints - 1
    return None


def _infer_ball_keys(layout: str, num_joints: int, ball_index: int, ball_key_joints):
    keys = ball_key_joints or _DEFAULT_BALL_KEYS.get(layout, [])
    filtered = [k for k in keys if 0 <= k < num_joints and k != ball_index]
    return filtered


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            x = torch.einsum("bctv,vw->bctw", x, adj)
        elif adj.dim() == 4:
            x = torch.einsum("bctv,btvw->bctw", x, adj)
        else:
            raise ValueError("adj must have shape [V,V] or [B,T,V,V]")
        return self.proj(x)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                padding=(4, 0),
                stride=(stride, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        res = 0 if self.residual is None else self.residual(x)
        x = self.gcn(x, adj)
        x = self.tcn(x)
        return self.relu(x + res)


class STGCNBackbone(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        in_channels: int = 3,
        layout: str = "coco17",
        edge_importance: bool = True,
        dropout: float = 0.1,
        enable_ball_dynamic: bool = True,
        ball_joint_index=None,
        ball_key_joints=None,
        init_sigma: float = 0.2,
    ):
        super().__init__()
        edges = _build_skeleton_edges(layout, num_joints)
        adj = torch.zeros(num_joints, num_joints)
        for i in range(num_joints):
            adj[i, i] = 1.0
        for i, j in edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        adj = _normalize_adjacency(adj)

        self.register_buffer("adj", adj)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.layers = nn.ModuleList(
            [
                STGCNBlock(in_channels, 64, stride=1, dropout=dropout, residual=False),
                STGCNBlock(64, 128, stride=2, dropout=dropout),
                STGCNBlock(128, 256, stride=2, dropout=dropout),
                STGCNBlock(256, 512, stride=2, dropout=dropout),
            ]
        )

        self.edge_importance = None
        if edge_importance:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones_like(adj)) for _ in self.layers]
            )

        self.ball_index = _infer_ball_index(layout, num_joints, ball_joint_index)
        self.ball_key_joints = _infer_ball_keys(
            layout, num_joints, self.ball_index, ball_key_joints
        )
        self.enable_ball_dynamic = (
            enable_ball_dynamic
            and self.ball_index is not None
            and len(self.ball_key_joints) > 0
            and in_channels >= 2
        )

        safe_sigma = max(init_sigma, 1e-4)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(safe_sigma)))

        self.feat_dim = 512
        self.fc = nn.Sequential()

    def _build_dynamic_adjacency(self, coords: torch.Tensor) -> torch.Tensor:
        if not self.enable_ball_dynamic:
            return None

        B, C, T, V = coords.shape
        if self.ball_index >= V:
            return None

        key_idx = torch.tensor(self.ball_key_joints, device=coords.device)
        ball_xy = coords[:, :2, :, self.ball_index].permute(0, 2, 1)  # [B, T, 2]
        joint_xy = coords[:, :2, :, key_idx].permute(0, 2, 3, 1)  # [B, T, K, 2]

        dist = torch.norm(ball_xy.unsqueeze(2) - joint_xy, dim=-1)  # [B, T, K]
        sigma = F.softplus(self.log_sigma) + 1e-4
        weights = torch.exp(-dist / sigma)

        if C >= 3:
            ball_conf = coords[:, 2, :, self.ball_index].clamp(min=0.0, max=1.0)
            weights = weights * ball_conf.unsqueeze(-1)

        adj = self.adj.view(1, 1, V, V).expand(B, T, V, V).clone()
        adj[:, :, self.ball_index, key_idx] = (
            adj[:, :, self.ball_index, key_idx] + weights
        )
        adj[:, :, key_idx, self.ball_index] = (
            adj[:, :, key_idx, self.ball_index] + weights
        )

        return _normalize_adjacency_dynamic(adj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, V]
        B, C, T, V = x.shape
        dynamic_adj = self._build_dynamic_adjacency(x)

        x = x.permute(0, 3, 1, 2).contiguous()  # [B, V, C, T]
        x = x.view(B, V * C, T)
        x = self.data_bn(x)
        x = x.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()  # [B, C, T, V]

        for idx, layer in enumerate(self.layers):
            if dynamic_adj is None:
                adj = self.adj
            else:
                adj = dynamic_adj
            if self.edge_importance is not None:
                if adj.dim() == 2:
                    adj = adj * self.edge_importance[idx]
                else:
                    adj = adj * self.edge_importance[idx].view(1, 1, V, V)
            x = layer(x, adj)

        x = x.mean(dim=-1)  # Uśrednij TYLKO węzły (stawy), zachowaj czas!
        x = x.permute(0, 2, 1)  # Zwróć [B, T=3, 512]
        return x
