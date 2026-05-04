"""
retrieval/mvit_retriever.py
===========================
MViT-v2-S visual feature extraction + FAISS nearest-neighbour retrieval
for RAG-ICL (Retrieval-Augmented In-Context Learning).
"""

import json
import numpy as np
import torch
from typing import List
from PIL import Image


class MViTRetriever:
    """
    Extracts MViT-v2-S features from PIL frame lists and queries FAISS.
    Uses diverse retrieval: one example per unique action class from top-50 candidates.
    """
    TARGET_FRAMES = 16
    FEAT_DIM = 768

    def __init__(self, index_path: str, meta_path: str, device: str = "cuda"):
        import faiss
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

        print("[MViTRetriever] Loading MViT-v2-S + FAISS index...")
        self.device = device
        weights = MViT_V2_S_Weights.DEFAULT
        model = mvit_v2_s(weights=weights)
        model.head = torch.nn.Identity()
        self.model = model.to(device).eval()
        self.transform = weights.transforms()
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.metadata = json.load(f)
        print(f"[MViTRetriever] Index has {self.index.ntotal} vectors.")

    @torch.no_grad()
    def _extract_feature(self, pil_frames: List[Image.Image]) -> np.ndarray:
        frames_np = np.stack([np.array(f.convert("RGB")) for f in pil_frames])
        T = len(frames_np)
        indices = np.linspace(0, T - 1, self.TARGET_FRAMES, dtype=int).tolist()
        sampled = frames_np[indices]
        video = torch.from_numpy(sampled).permute(0, 3, 1, 2).to(torch.uint8)
        inp = self.transform(video).unsqueeze(0).to(self.device)
        feat = self.model(inp).cpu().numpy().astype(np.float32).flatten()
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-8)

    def retrieve(self, live_frames: List[Image.Image], k: int = 3) -> str:
        """
        Diverse retrieval: one example per unique action class from top-50 candidates.
        Prevents majority-class bias (Tackling dominating all retrievals).
        """
        feat = self._extract_feature(live_frames).reshape(1, -1).astype(np.float32)
        n_candidates = min(self.index.ntotal, 50)
        distances, indices = self.index.search(feat, n_candidates)

        seen_actions = {}
        for dist, idx in zip(distances[0], indices[0]):
            meta = self.metadata.get(str(idx), {})
            action = meta.get("action", "Unknown")
            if action not in seen_actions:
                seen_actions[action] = (float(dist), meta)
            if len(seen_actions) >= k:
                break

        examples_str = ""
        for rank, (action, (dist, meta)) in enumerate(seen_actions.items(), 1):
            sev = meta.get("severity", "Unknown")
            examples_str += (
                f"PRECEDENT {rank} (visual distance={dist:.3f}):\n"
                f'  Decision: {{"action": "{action}", "severity": "{sev}"}}\n\n'
            )
        return examples_str.strip()

    def retrieve_hard_negatives(self, live_frames: List[Image.Image], predicted_action: str, k: int = 3) -> str:
        '''Retrieves visually similar clips of the SAME action, but ensures diverse severities are shown.'''
        feat = self._extract_feature(live_frames).reshape(1, -1).astype(np.float32)
        # Search deeper to find enough boundary cases
        distances, indices = self.index.search(feat, min(self.index.ntotal, 100))
        
        seen_severities = set()
        hard_negatives = []
        
        for dist, idx in zip(distances[0], indices[0]):
            meta = self.metadata.get(str(idx), {})
            if meta.get("action") == predicted_action:
                sev = meta.get("severity")
                if sev not in seen_severities:
                    seen_severities.add(sev)
                    hard_negatives.append((dist, meta))
            if len(seen_severities) >= k:
                break
                
        examples_str = ""
        for rank, (dist, meta) in enumerate(hard_negatives, 1):
            action, sev = meta.get("action", "Unknown"), meta.get("severity", "Unknown")
            examples_str += (
                f"BOUNDARY PRECEDENT {rank} (visual distance={dist:.3f}):\n"
                f'  Decision: {{"action": "{action}", "severity": "{sev}"}}\n\n'
            )
        return examples_str.strip()
