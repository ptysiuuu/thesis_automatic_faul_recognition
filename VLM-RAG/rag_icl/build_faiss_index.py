"""
build_faiss_index.py
====================
Builds a FAISS index of MViT-v2-S visual features from the Train split.

Run ONCE before evaluation. Produces:
  - train_mvit_features.index   : FAISS flat L2 index (~2300 vectors × 768 dims)
  - train_mvit_metadata.json    : maps FAISS index → action label + severity

Usage:
  python build_faiss_index.py \
    --hdf5_path   /net/tscratch/people/plgaszos/SoccerNet_HDF5/Train.hdf5 \
    --annotations /net/tscratch/people/plgaszos/SoccerNet_Data/Train/annotations.json \
    --output_dir  /net/tscratch/people/plgaszos/vlm_rag_icl

Dependencies:
  pip install faiss-gpu torch torchvision h5py tqdm
  (faiss-cpu also works, index building is fast enough)
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


# ---------------------------------------------------------------------------
# Label mappings — must match evaluate_vlm.py
# ---------------------------------------------------------------------------

ACTION_CLASSES = [
    "Tackling", "Standing tackling", "High leg", "Holding",
    "Pushing", "Elbowing", "Challenge", "Dive",
]

# Raw severity string → human-readable card label
SEVERITY_MAP = {
    "":    "No offence",
    "1.0": "No card",
    "3.0": "Yellow card",
    "5.0": "Red card",
}

SKIP_ACTIONS   = {"Dont know", ""}
SKIP_SEVERITIES = {"2.0", "4.0"}   # ambiguous borderline labels


# ---------------------------------------------------------------------------
# MViT feature extractor
# ---------------------------------------------------------------------------

class MViTFeatureExtractor:
    """
    MViT-v2-S with classification head removed.
    Input : HDF5 raw frames [T, H, W, C] uint8
    Output: 1-D numpy float32 vector of size 768
    """

    FEAT_DIM = 768
    TARGET_FRAMES = 16

    def __init__(self, device: str = "cuda"):
        self.device = device
        print("[MViT] Loading MViT-v2-S feature extractor...")
        weights = MViT_V2_S_Weights.DEFAULT
        model = mvit_v2_s(weights=weights)
        # Replace head with identity so forward() returns the 768-dim pooled vector
        model.head = torch.nn.Identity()
        self.model = model.to(device).eval()
        self.transform = weights.transforms()
        print(f"[MViT] Ready. Feature dim = {self.FEAT_DIM}")

    @torch.no_grad()
    def extract(self, frames_np: np.ndarray) -> np.ndarray:
        """
        frames_np : [T, H, W, C] uint8  (raw HDF5 tensor)
        Returns   : [768] float32
        """
        # [T, H, W, C] → [C, T, H, W]
        video = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()

        # Resample to TARGET_FRAMES along time axis
        T = video.shape[1]
        if T != self.TARGET_FRAMES:
            video = F.interpolate(
                video.unsqueeze(0),   # [1, C, T, H, W]
                size=(self.TARGET_FRAMES, video.shape[2], video.shape[3]),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)             # [C, 16, H, W]

        # MViT transforms expect uint8 [C, T, H, W]
        video_uint8 = video.clamp(0, 255).to(torch.uint8)
        input_tensor = self.transform(video_uint8).unsqueeze(0).to(self.device)

        feat = self.model(input_tensor)  # [1, 768]
        return feat.cpu().numpy().astype(np.float32).flatten()


# ---------------------------------------------------------------------------
# Main index builder
# ---------------------------------------------------------------------------

def build_index(args):
    import h5py
    import faiss

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "train_mvit_features.index"
    meta_path  = output_dir / "train_mvit_metadata.json"

    extractor = MViTFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu")

    with open(args.annotations) as f:
        annotations = json.load(f)["Actions"]

    # FAISS flat L2 index (exact search, fast enough for ~2300 vectors)
    index    = faiss.IndexFlatL2(MViTFeatureExtractor.FEAT_DIM)
    metadata = {}
    idx_counter = 0
    n_skipped   = 0

    with h5py.File(args.hdf5_path, "r", swmr=True) as hdf5:
        for action_id, action_data in tqdm(annotations.items(), desc="Building FAISS"):
            action_class   = action_data.get("Action class", "")
            offence_class  = action_data.get("Offence", "")
            severity_str   = action_data.get("Severity", "")

            # Skip ambiguous / unknown labels
            if action_class in SKIP_ACTIONS:
                n_skipped += 1
                continue
            if severity_str in SKIP_SEVERITIES:
                n_skipped += 1
                continue

            # Map severity to readable string
            # Handle "No offence" case: Offence="" means no card
            if offence_class in ("No offence", "No Offence", ""):
                severity_label = "No offence"
            else:
                severity_label = SEVERITY_MAP.get(severity_str, "No card")

            if action_class not in ACTION_CLASSES:
                n_skipped += 1
                continue

            # Load frames from the first (live) clip
            clips = action_data.get("Clips", [])
            if not clips:
                n_skipped += 1
                continue

            clip_name = clips[0]["Url"].split("/")[-1].replace(".mp4", "")
            hdf5_key  = f"action_{action_id}/{clip_name}"

            if hdf5_key not in hdf5:
                n_skipped += 1
                continue

            frames_np = hdf5[hdf5_key][:]   # [T, H, W, C] uint8
            if len(frames_np) < 4:
                n_skipped += 1
                continue

            try:
                feat = extractor.extract(frames_np)   # [768]
            except Exception as e:
                print(f"  Error extracting {hdf5_key}: {e}")
                n_skipped += 1
                continue

            # L2-normalize before storing so cosine similarity = dot product
            norm = np.linalg.norm(feat)
            if norm > 1e-8:
                feat = feat / norm

            index.add(feat.reshape(1, -1))

            metadata[str(idx_counter)] = {
                "action":    action_class,
                "severity":  severity_label,
                "action_id": action_id,
            }
            idx_counter += 1

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[FAISS] Indexed {idx_counter} samples ({n_skipped} skipped)")
    print(f"[FAISS] Index  → {index_path}")
    print(f"[FAISS] Meta   → {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path",   required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output_dir",  required=True)
    args = parser.parse_args()
    build_index(args)
