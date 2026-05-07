import torch
import torch.nn as nn
from typing import List

# ---------------------------------------------------------------------------
# Fixed class-level prompts (no sample-specific labels)
# ---------------------------------------------------------------------------

ACTION_PROMPTS = [
    "a player slides or extends their leg to dispossess an opponent",
    "a player uses their standing leg to block the ball",
    "a player raises their foot dangerously near an opponent's head",
    "a player grabs or restrains an opponent with their arm",
    "a player uses their arm or body to push an opponent",
    "a player strikes an opponent with their elbow",
    "two players physically challenge each other for the ball",
    "a player falls to the ground to deceive the referee",
]

SEVERITY_PROMPTS = [
    "fair challenge, no foul committed, legal contact",
    "minor foul, careless but not dangerous, no card needed",
    "reckless challenge showing disregard for opponent, yellow card",
    "excessive force endangering opponent safety, red card",
]


# ---------------------------------------------------------------------------
# CLIP text encoder wrapper
# ---------------------------------------------------------------------------


class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP ViT-B/16 text encoder.
    Encodes a list of strings -> [N, 512].
    """

    def __init__(self):
        super().__init__()
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        model, _ = clip.load("ViT-B/16", device="cpu")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.tokenize = clip.tokenize

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenize(texts, truncate=True)
        device = next(self.model.parameters()).device
        feats = self.model.encode_text(tokens.to(device))
        return feats.float()  # [N, 512]


# ---------------------------------------------------------------------------
# Cross-attention bridge
# ---------------------------------------------------------------------------


class TextConditionedBridge(nn.Module):
    """
    Lightweight cross-attention that conditions visual features on
    fixed class-level text embeddings.

    visual : [B*V, 768]   - TAdaFormer output (one vector per view)
    Output : [B*V, 768]   - same shape, text-conditioned
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_dim = visual_dim

        encoder = CLIPTextEncoder()
        with torch.no_grad():
            action_embs = encoder(ACTION_PROMPTS)
            severity_embs = encoder(SEVERITY_PROMPTS)

        self.register_buffer("action_embs", action_embs, persistent=True)
        self.register_buffer("severity_embs", severity_embs, persistent=True)

        self.action_proj = nn.Linear(text_dim, visual_dim)
        self.severity_proj = nn.Linear(text_dim, visual_dim)

        self.action_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.severity_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(visual_dim)
        self.dropout = nn.Dropout(dropout)

        # Learned gate: starts at 0 (sigmoid=0.5) for equal action/severity mix
        self.gate = nn.Parameter(torch.zeros(1))

        # Small init to avoid disrupting pretrained backbone
        nn.init.xavier_uniform_(self.action_proj.weight, gain=0.1)
        nn.init.zeros_(self.action_proj.bias)
        nn.init.xavier_uniform_(self.severity_proj.weight, gain=0.1)
        nn.init.zeros_(self.severity_proj.bias)

    def forward(
        self, visual: torch.Tensor, num_views: int, return_attn: bool = False
    ) -> torch.Tensor:
        """
        visual   : [B*V, 768]
        num_views: V

        Returns  : [B*V, 768] (optionally attention weights)
        """
        BV, _ = visual.shape
        visual_q = visual.unsqueeze(1)  # [B*V, 1, 768]

        action_kv = self.action_proj(self.action_embs)  # [8, 768]
        severity_kv = self.severity_proj(self.severity_embs)  # [4, 768]

        action_kv = action_kv.unsqueeze(0).expand(BV, -1, -1)
        severity_kv = severity_kv.unsqueeze(0).expand(BV, -1, -1)

        attn_action, weights_action = self.action_attn(
            query=visual_q, key=action_kv, value=action_kv
        )
        attn_severity, weights_severity = self.severity_attn(
            query=visual_q, key=severity_kv, value=severity_kv
        )

        gate = torch.sigmoid(self.gate)
        combined = gate * attn_action + (1.0 - gate) * attn_severity
        combined = combined.squeeze(1)  # [B*V, 768]

        out = self.norm(visual + self.dropout(combined))
        if return_attn:
            return out, weights_action, weights_severity
        return out
