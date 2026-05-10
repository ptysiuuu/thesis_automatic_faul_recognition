"""
utils/frames_local.py
=====================
Frame extraction from raw SoccerNet .mp4 clip files.

Used when running locally without the HDF5 pre-extracted frames
(which are only on the Athena cluster). Reads directly from the
standard SoccerNet-MVFoul directory structure:

    <data_root>/
    ├── Train/
    │   ├── annotations.json
    │   └── action_0/
    │       ├── clip_0.mp4
    │       ├── clip_1.mp4
    │       └── clip_2.mp4   (optional)
    ├── Valid/
    │   └── ...
    └── Test/
        └── ...

Drop-in replacement for the HDF5-based functions in utils/frames.py.
All downstream strategies (cos_two_stage, static_few_shot, etc.) work
unchanged — they only see List[List[PIL.Image]].

Dependencies (install with pip):
    pip install opencv-python-headless pillow

Note: OpenCV is used for video decoding. If unavailable, falls back
to imageio (slower but pure Python).
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional
from PIL import Image


# ---------------------------------------------------------------------------
# Core video → frames extraction
# ---------------------------------------------------------------------------

def _read_frames_cv2(video_path: str, n_frames: int = 4) -> List[Image.Image]:
    """Extract n evenly-spaced frames using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # BGR → RGB
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames


def _read_frames_imageio(video_path: str, n_frames: int = 4) -> List[Image.Image]:
    """Fallback: extract frames using imageio (slower)."""
    import imageio.v3 as iio

    try:
        video = iio.imread(video_path, plugin="pyav")
    except Exception:
        try:
            video = iio.imread(video_path)
        except Exception:
            return []

    total = len(video)
    if total < 2:
        return []

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    return [Image.fromarray(video[i]) for i in indices]


def read_clip_frames(video_path: str, n_frames: int = 4) -> List[Image.Image]:
    """
    Read n evenly-spaced frames from a video file.
    Tries OpenCV first, falls back to imageio.
    """
    if not os.path.exists(video_path):
        return []
    try:
        import cv2
        frames = _read_frames_cv2(video_path, n_frames)
        if frames:
            return frames
    except ImportError:
        pass

    return _read_frames_imageio(video_path, n_frames)


def read_clip_frames_weighted(video_path: str, n_frames: int = 8) -> List[Image.Image]:
    """
    Extract frames weighted toward clip center (contact moment).
    Same logic as extract_keyframes_weighted in frames.py but for mp4 files.
    """
    import cv2

    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if T < 4:
        return read_clip_frames(video_path, min(n_frames, T))

    center = T // 2
    dense_start = max(0, center - T // 5)
    dense_end   = min(T - 1, center + T // 5)
    n_dense = max(1, n_frames - 4)
    n_edge  = (n_frames - n_dense) // 2

    early  = np.linspace(0, dense_start, n_edge + 1, dtype=int)[:-1]
    dense  = np.linspace(dense_start, dense_end, n_dense, dtype=int)
    late   = np.linspace(dense_end, T - 1, n_edge + 1, dtype=int)[1:]
    indices = np.unique(np.concatenate([early, dense, late]))[:n_frames]

    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in sorted(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# SoccerNet directory structure helpers
# ---------------------------------------------------------------------------

def get_clip_path(data_root: str, split: str, action_id: str, clip_name: str) -> str:
    """
    Build full path to a clip file.

    SoccerNet structure:
        <data_root>/<split>/action_<action_id>/<clip_name>

    clip_name examples: "clip_0.mp4", "clip_1.mp4"
    """
    return os.path.join(data_root, split, f"action_{action_id}", clip_name)


def load_annotations_local(data_root: str, split: str) -> dict:
    """
    Load annotations.json from the SoccerNet directory structure.
    Returns same format as the HDF5-based load_annotations():
        {action_id: {"action": int, "severity": int, "clips": [str]}}

    clip strings are bare filenames like "clip_0.mp4"
    (not URLs — stripped for local use).
    """
    import json

    ann_path = os.path.join(data_root, split, "annotations.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotations not found: {ann_path}")

    with open(ann_path) as f:
        data = json.load(f)

    from .constants import ACTION_TO_IDX, OFFENCE_SEVERITY_MAP

    samples = {}
    for action_id, action_data in data["Actions"].items():
        action_class  = action_data.get("Action class", "")
        offence_class = action_data.get("Offence", "")
        severity_class = action_data.get("Severity", "")

        if action_class in {"Dont know", ""}:
            continue
        if offence_class in {"Between", ""} and action_class != "Dive":
            continue
        if severity_class in {"2.0", "4.0"} and action_class != "Dive" \
                and offence_class not in ("No offence", "No Offence"):
            continue

        if offence_class in {"Between", ""}:
            offence_class = "Offence"
        if severity_class in {"2.0", "4.0"}:
            severity_class = "1.0"

        key = (offence_class, severity_class)
        if key in OFFENCE_SEVERITY_MAP:
            severity_idx = OFFENCE_SEVERITY_MAP[key]
        elif offence_class in ("No offence", "No Offence"):
            severity_idx = 0
        else:
            continue

        action_idx = ACTION_TO_IDX.get(action_class, -1)
        if action_idx == -1:
            continue

        # Build clip filenames — SoccerNet clips are clip_0.mp4, clip_1.mp4, etc.
        clips_data = action_data.get("Clips", [])
        clip_names = []
        for c in clips_data:
            url = c.get("Url", "")
            # Extract just the filename: "...action_0/clip_0.mp4" → "clip_0.mp4"
            clip_file = url.split("/")[-1] if url else ""
            if clip_file:
                clip_names.append(clip_file)

        # Fallback: if no URLs, assume standard naming
        if not clip_names:
            clip_names = [f"clip_{i}.mp4" for i in range(len(clips_data))]

        samples[action_id] = {
            "action":   action_idx,
            "severity": severity_idx,
            "clips":    clip_names,
        }

    return samples


def extract_all_views_local(
    data_root: str,
    split: str,
    action_id: str,
    clip_names: List[str],
    n_frames: int = 4,
    weighted: bool = False,
    max_views: int = 4,
) -> List[List[Image.Image]]:
    """
    Extract frames for all views of an action from mp4 files.
    Drop-in replacement for extract_all_views() from frames.py.

    Returns list of frame lists (one per view), same as HDF5 version.
    """
    extractor = read_clip_frames_weighted if weighted else read_clip_frames
    frames_per_view = []

    for clip_name in clip_names[:max_views]:
        clip_path = get_clip_path(data_root, split, action_id, clip_name)
        frames = extractor(clip_path, n_frames)
        if frames:
            frames_per_view.append(frames)

    return frames_per_view
