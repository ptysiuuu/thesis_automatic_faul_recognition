"""
retrieval/medoid_cache.py
=========================
Builds and loads a cache of medoid training examples per (action × severity) class.
Each medoid is the most prototypical training clip for its class, stored as
base64 JPEG frames for injection into VLM prompts.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from io import BytesIO
from typing import List, Tuple
from PIL import Image

from ..utils.annotations import load_annotations
from ..utils.constants import ACTION_CLASSES, SEVERITY_CLASSES, CONFUSABLE_ACTIONS


def build_medoid_cache(train_hdf5_path: str, train_annotations_path: str,
                       faiss_index_path: str, faiss_meta_path: str,
                       output_path: str, n_frames_per_example: int = 2):
    """
    For each (action, severity) class combination, find the training sample
    whose MViT feature is closest to the class centroid (medoid).
    Saves N frames as base64 JPEGs.
    """
    import faiss, h5py

    FEAT_DIM = 768
    print("[MedoidCache] Loading FAISS index and metadata...")
    index = faiss.read_index(faiss_index_path)
    with open(faiss_meta_path) as f:
        metadata = json.load(f)

    # Group FAISS indices by (action, severity)
    groups = defaultdict(list)
    for idx_str, meta in metadata.items():
        key = f"{meta['action']}|{meta['severity']}"
        groups[key].append(int(idx_str))

    print(f"[MedoidCache] {len(groups)} (action, severity) groups:")
    for k, v in sorted(groups.items()):
        print(f"  {k}: {len(v)} samples")

    # Reconstruct feature vectors
    all_feats = np.zeros((index.ntotal, FEAT_DIM), dtype=np.float32)
    for i in range(index.ntotal):
        faiss.downcast_index(index).reconstruct(i, all_feats[i])

    # Find medoid per group
    medoid_faiss_indices = {}
    for key, indices in groups.items():
        if len(indices) == 1:
            medoid_faiss_indices[key] = indices[0]
            continue
        feats = all_feats[indices]
        dists = np.sum((feats[:, None] - feats[None, :]) ** 2, axis=-1)
        medoid_faiss_indices[key] = indices[np.argmin(dists.sum(axis=1))]

    # Load frames for each medoid
    train_samples = load_annotations(train_annotations_path)
    faiss_to_action = {int(idx_str): meta["action_id"]
                       for idx_str, meta in metadata.items()}

    cache = {}
    with h5py.File(train_hdf5_path, "r", swmr=True) as hdf5:
        for key, faiss_idx in medoid_faiss_indices.items():
            action_id = faiss_to_action.get(faiss_idx)
            if not action_id or action_id not in train_samples:
                continue
            sample = train_samples[action_id]

            frames_b64 = []
            for clip_raw in sample["clips"][:1]:
                clip_key = clip_raw.replace(".mp4", "")
                hdf5_key = f"action_{action_id}/{clip_key}"
                if hdf5_key not in hdf5:
                    continue
                frames_np = hdf5[hdf5_key][:]
                T = len(frames_np)
                if T < 2:
                    continue
                indices = np.linspace(0, T - 1, n_frames_per_example, dtype=int)
                for idx in indices:
                    img = Image.fromarray(frames_np[idx])
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    import base64
                    frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

            if frames_b64:
                action_str, severity_str = key.split("|")
                cache[key] = {
                    "action":     action_str,
                    "severity":   severity_str,
                    "action_id":  action_id,
                    "frames_b64": frames_b64,
                }
                print(f"  ✓ {key} → action_id={action_id}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache, f)
    print(f"[MedoidCache] Saved {len(cache)} entries → {output_path}")
    return cache


def load_medoid_cache(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_examples_text(medoid_cache: dict,
                         n_per_class: int = 1) -> Tuple[str, List[Image.Image]]:
    """All medoid examples as text + PIL images."""
    import base64
    examples_text = ""
    example_images = []
    for i, (key, entry) in enumerate(sorted(medoid_cache.items()), 1):
        action, severity = entry["action"], entry["severity"]
        examples_text += (f"EXAMPLE {i} — {action} / {severity}:\n"
                          f'  Decision: {{"action": "{action}", "severity": "{severity}"}}\n\n')
        for b64 in entry["frames_b64"][:n_per_class]:
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            example_images.append(img)
    return examples_text.strip(), example_images


def build_targeted_examples(medoid_cache: dict,
                             predicted_action: str,
                             n_same_action: int = 2,
                             n_confusable: int = 1) -> Tuple[str, List[Image.Image]]:
    """
    Build targeted examples for a predicted action:
    - n_same_action examples of the predicted action (different severities)
    - n_confusable examples from the most easily confused action

    This is the key fix for data_driven: instead of all examples every time,
    show only contextually relevant ones.
    """
    import base64
    selected = []

    # Same action, different severities
    for key, entry in sorted(medoid_cache.items()):
        if entry["action"] == predicted_action and len(selected) < n_same_action:
            selected.append(entry)

    # Confusable action
    confusable = CONFUSABLE_ACTIONS.get(predicted_action, [])
    for conf_action in confusable:
        for key, entry in sorted(medoid_cache.items()):
            if entry["action"] == conf_action and len(selected) < n_same_action + n_confusable:
                selected.append(entry)
                break

    examples_text = ""
    example_images = []
    for i, entry in enumerate(selected, 1):
        action, severity = entry["action"], entry["severity"]
        examples_text += (f"EXAMPLE {i} — {action} / {severity}:\n"
                          f'  Decision: {{"action": "{action}", "severity": "{severity}"}}\n\n')
        for b64 in entry["frames_b64"][:1]:
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            example_images.append(img)

    return examples_text.strip(), example_images


def build_severity_examples(medoid_cache: dict,
                             action: str) -> Tuple[str, List[Image.Image]]:
    """Examples for a specific action, covering all severity levels."""
    import base64
    examples_text = ""
    example_images = []
    rank = 1
    for key, entry in sorted(medoid_cache.items()):
        if entry["action"] != action:
            continue
        severity = entry["severity"]
        examples_text += (f"EXAMPLE {rank} — {action} / {severity}:\n"
                          f'  Decision: {{"action": "{action}", "severity": "{severity}"}}\n\n')
        for b64 in entry["frames_b64"][:1]:
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            example_images.append(img)
        rank += 1
    return examples_text.strip(), example_images
