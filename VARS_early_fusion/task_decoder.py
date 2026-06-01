import torch
import torch.nn as nn


class _TaskQueryDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm_self = nn.LayerNorm(dim)
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # query: [B, 1, D], tokens: [B, N, D]
        q_norm = self.norm_self(query)
        q_sa, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        query = query + q_sa

        q_norm = self.norm_cross_q(query)
        kv_norm = self.norm_cross_kv(tokens)
        q_ca, _ = self.cross_attn(
            q_norm,
            kv_norm,
            kv_norm,
            key_padding_mask=token_mask,
            need_weights=False,
        )
        query = query + q_ca

        query = query + self.ffn(self.norm_ffn(query))
        return query


class TaskQueryDecoder(nn.Module):
    """
    Dual-branch task-specific query decoder.

    Returns:
        q_action: [B, D]
        q_severity: [B, D]
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.action_query = nn.Parameter(torch.empty(1, 1, dim))
        self.severity_query = nn.Parameter(torch.empty(1, 1, dim))

        self.action_layers = nn.ModuleList(
            [_TaskQueryDecoderLayer(dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.severity_layers = nn.ModuleList(
            [_TaskQueryDecoderLayer(dim, num_heads, dropout) for _ in range(num_layers)]
        )

        nn.init.uniform_(self.action_query, -0.1, 0.1)
        nn.init.uniform_(self.severity_query, -0.1, 0.1)

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor = None,
    ) -> tuple:
        if tokens.dim() != 3:
            raise ValueError("tokens must be [B, N, D]")
        if tokens.shape[-1] != self.dim:
            raise ValueError("tokens last dimension must match dim")

        batch_size = tokens.shape[0]
        q_act = self.action_query.expand(batch_size, -1, -1)
        q_sev = self.severity_query.expand(batch_size, -1, -1)

        for layer in self.action_layers:
            q_act = layer(q_act, tokens, token_mask=token_mask)
        for layer in self.severity_layers:
            q_sev = layer(q_sev, tokens, token_mask=token_mask)

        return q_act.squeeze(1), q_sev.squeeze(1)
