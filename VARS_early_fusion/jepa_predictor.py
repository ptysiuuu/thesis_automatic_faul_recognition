import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """
    Cross-attention predictor for JEPA self-supervised objectives.

    Target queries (learnable position embeddings for masked slots) attend into
    student context tokens to predict the teacher's latent representations at
    the masked positions.

    Args:
        dim:        Feature dimensionality D.
        num_layers: Number of TransformerDecoderLayer stacks (default 2).
        num_heads:  Attention heads — must divide dim evenly (default 8).
        dropout:    Dropout probability (default 0.1).

    Inputs:
        context_tokens : [B, N_ctx, D]  — student aggregator output tokens
        target_queries : [B, N_tgt, D]  — position embeddings for masked slots

    Returns:
        [B, N_tgt, D] — predicted teacher latents at masked positions
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_queries: torch.Tensor,
    ) -> torch.Tensor:
        x = target_queries
        for layer in self.layers:
            x = layer(x, context_tokens)
        return self.norm(x)
