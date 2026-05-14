import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPSeverityLoss(nn.Module):
    def __init__(self, embeddings_path: str, temperature: float = 0.07):
        super().__init__()
        data = torch.load(embeddings_path, map_location="cpu")
        if isinstance(data, dict) and "embeddings" in data:
            embeddings = data["embeddings"]
        else:
            embeddings = data

        if embeddings.ndim != 2 or embeddings.shape[0] != 4:
            raise ValueError(
                "CLIP embeddings must be a [4, 512] tensor or a dict with key 'embeddings'."
            )

        embeddings = F.normalize(embeddings.float(), dim=-1)
        self.register_buffer("text_embeddings", embeddings)
        self.temperature = temperature

    def forward(
        self, visual_proj: torch.Tensor, labels_int: torch.Tensor
    ) -> torch.Tensor:
        visual_proj = F.normalize(visual_proj.float(), dim=-1)
        logits = visual_proj @ self.text_embeddings.T
        logits = logits / self.temperature
        return F.cross_entropy(logits, labels_int.long())
