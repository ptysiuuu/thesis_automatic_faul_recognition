import torch
import torch.nn as nn
from typing import List

# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

ACTION_DESCRIPTIONS = {
    "Tackling": "player slides to take the ball",
    "Standing tackling": "player uses standing leg to take the ball",
    "High leg": "player raises leg dangerously near opponent",
    "Holding": "player grabs or restrains opponent",
    "Pushing": "player shoves opponent with arm or body",
    "Elbowing": "player strikes opponent with elbow",
    "Challenge": "players compete for ball with physical contact",
    "Dive": "player exaggerates fall to deceive referee",
}

BODYPART_PHRASES = {
    "Upper body": "upper body contact",
    "Under body": "lower body contact",
}

UPPER_BODY_PHRASES = {
    "Use of shoulder": "using shoulder",
    "Use of arm": "using arm",
    "Use of elbow": "using elbow",
    "Use of hand": "using hand",
    "Use of head": "using head",
    "Use of chest": "using chest",
}


def build_prompt(
    action_class: str,
    contact: str,
    bodypart: str,
    upper_body_part: str,
    try_to_play: str,
    touch_ball: str,
) -> str:
    """
    Build a natural language prompt from annotation fields.

    Example output:
    "Challenge with upper body contact using shoulder, player tried to play the ball"
    """
    parts = []

    base = ACTION_DESCRIPTIONS.get(action_class, action_class.lower())
    parts.append(base)

    if contact == "With contact":
        bp = BODYPART_PHRASES.get(bodypart, "")
        ubp = UPPER_BODY_PHRASES.get(upper_body_part, "")
        if bp and ubp:
            parts.append(f"with {bp} {ubp}")
        elif bp:
            parts.append(f"with {bp}")

    if try_to_play == "Yes":
        if touch_ball == "Yes":
            parts.append("player tried to play ball and touched it")
        elif touch_ball == "No":
            parts.append("player tried to play ball but did not touch it")
        else:
            parts.append("player tried to play the ball")
    elif try_to_play == "No":
        parts.append("player made no attempt to play the ball")

    return ", ".join(parts)


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
    per-sample text embeddings.

    visual : [B*V, 768]   - TAdaFormer output (one vector per view)
    text   : [B, 512]     - CLIP embedding of per-sample prompt

    Output : [B*V, 768]   - same shape, text-conditioned.
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

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.LayerNorm(visual_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(visual_dim)
        self.dropout = nn.Dropout(dropout)

        # Learned gate: starts at 0 so bridge has zero effect at init
        self.gate = nn.Parameter(torch.zeros(1))

        # Small init to avoid disrupting pretrained backbone
        nn.init.xavier_uniform_(self.text_proj[0].weight, gain=0.1)
        nn.init.zeros_(self.text_proj[0].bias)

    def forward(
        self, visual: torch.Tensor, text_emb: torch.Tensor, num_views: int
    ) -> torch.Tensor:
        """
        visual   : [B*V, 768]
        text_emb : [B, 512]   - one prompt per action
        num_views: V

        Returns  : [B*V, 768]
        """
        BV, D = visual.shape
        B = BV // num_views

        text_proj = self.text_proj(text_emb)  # [B, 768]

        # Repeat text V times to align with flattened views
        text_kv = text_proj.unsqueeze(1)  # [B, 1, 768]
        text_kv = text_kv.repeat_interleave(num_views, dim=0)  # [B*V, 1, 768]

        visual_q = visual.unsqueeze(1)  # [B*V, 1, 768]
        attended, _ = self.cross_attn(query=visual_q, key=text_kv, value=text_kv)
        attended = attended.squeeze(1)  # [B*V, 768]

        gate = torch.sigmoid(self.gate)
        out = self.norm(visual + gate * self.dropout(attended))
        return out
