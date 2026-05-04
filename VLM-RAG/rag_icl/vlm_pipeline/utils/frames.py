"""
utils/frames.py
===============
Frame extraction from HDF5 and key frame selection utilities.
"""

import numpy as np
from typing import List, Tuple
from PIL import Image


def extract_keyframes(hdf5_file, action_key: str, clip_key: str,
                      n_frames: int = 4) -> List[Image.Image]:
    """Extract N evenly-spaced frames from HDF5 clip."""
    key = f"{action_key}/{clip_key}"
    if key not in hdf5_file:
        return []
    frames_np = hdf5_file[key][:]
    total = len(frames_np)
    if total < 2:
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    return [Image.fromarray(frames_np[i]) for i in indices]


def extract_keyframes_weighted(hdf5_file, action_key: str, clip_key: str,
                               n_frames: int = 8) -> List[Image.Image]:
    """
    Extract frames weighted toward clip center (where contact occurs in SoccerNet).
    2 frames from early context, 4 from dense center window, 2 from late context.
    Provides more candidate frames for cos_two_stage frame selection.
    """
    key = f"{action_key}/{clip_key}"
    if key not in hdf5_file:
        return []
    frames_np = hdf5_file[key][:]
    T = len(frames_np)
    if T < 4:
        indices = np.linspace(0, T - 1, min(n_frames, T), dtype=int)
        return [Image.fromarray(frames_np[i]) for i in indices]

    center     = T // 2
    dense_start = max(0, center - T // 5)
    dense_end   = min(T - 1, center + T // 5)

    n_dense = max(1, n_frames - 4)
    n_edge  = (n_frames - n_dense) // 2

    early  = np.linspace(0, dense_start, n_edge + 1, dtype=int)[:-1]
    dense  = np.linspace(dense_start, dense_end, n_dense, dtype=int)
    late   = np.linspace(dense_end, T - 1, n_edge + 1, dtype=int)[1:]

    indices = np.unique(np.concatenate([early, dense, late]))[:n_frames]
    return [Image.fromarray(frames_np[i]) for i in sorted(indices)]


def extract_all_views(hdf5_file, action_key: str, clips: list,
                      n_frames: int = 4,
                      weighted: bool = False,
                      max_views: int = 4) -> List[List[Image.Image]]:
    """
    Extract frames for all views of an action.
    Returns list of frame lists (one per view).
    """
    extractor = extract_keyframes_weighted if weighted else extract_keyframes
    frames_per_view = []
    for clip_raw in clips[:max_views]:
        clip_key = clip_raw.replace(".mp4", "")
        frames = extractor(hdf5_file, action_key, clip_key, n_frames=n_frames)
        if frames:
            frames_per_view.append(frames)
    return frames_per_view


def parse_key_frames(text: str, n_views: int, frames_per_view: int) -> List[int]:
    """
    Parse the CoS frame selection response.
    Returns list of length n_views with selected frame index per view.
    Falls back to middle frame on parse failure.
    """
    import re, json
    default = frames_per_view // 2
    result  = [default] * n_views

    text = re.sub(r"```json\s*|\s*```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return result
    try:
        data = json.loads(match.group())
    except Exception:
        return result

    view_labels = ["Live camera"] + [f"Replay {i}" for i in range(1, n_views)]
    for v_idx, label in enumerate(view_labels):
        val = data.get(label, default)
        try:
            idx = int(val)
            result[v_idx] = max(0, min(frames_per_view - 1, idx))
        except (TypeError, ValueError):
            result[v_idx] = default
    return result


def select_key_frames(frames_per_view_list: List[List[Image.Image]],
                      key_frame_indices: List[int],
                      context_window: int = 1) -> List[List[Image.Image]]:
    """
    For each view, return key frame ± context_window adjacent frames.
    context_window=1 → up to 3 frames per view (key-1, key, key+1).
    """
    selected = []
    for v_idx, frames in enumerate(frames_per_view_list):
        key = key_frame_indices[v_idx] if v_idx < len(key_frame_indices) else len(frames) // 2
        lo = max(0, key - context_window)
        hi = min(len(frames) - 1, key + context_window)
        selected.append(frames[lo:hi + 1])
    return selected


def format_selected_frame_info(key_frame_indices: List[int], n_views: int) -> str:
    """Human-readable summary of which frames were selected."""
    view_labels = ["Live camera"] + [f"Replay {i}" for i in range(1, n_views)]
    lines = []
    for v_idx, label in enumerate(view_labels[:n_views]):
        idx = key_frame_indices[v_idx] if v_idx < len(key_frame_indices) else "?"
        lines.append(f"  {label}: frame {idx}")
    return "\n".join(lines)

import cv2

def extract_flow_keyframe(hdf5_file, action_key: str, clip_key: str, context_window: int = 1) -> List[Image.Image]:
    '''Finds the contact moment using OpenCV peak pixel motion, bypassing the VLM.'''
    key = f"{action_key}/{clip_key}"
    if key not in hdf5_file: return []
    
    frames_np = hdf5_file[key][:]
    T = len(frames_np)
    if T < 3:
        return [Image.fromarray(f) for f in frames_np]
        
    # Fast frame differencing to find peak motion spike (impact)
    motion_scores = []
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_np]
    for i in range(T - 1):
        diff = cv2.absdiff(grays[i], grays[i+1])
        motion_scores.append(np.sum(diff))
        
    # Peak motion usually occurs right AT the contact frame
    peak_idx = int(np.argmax(motion_scores)) + 1
    peak_idx = min(peak_idx, T - 1)
    
    lo = max(0, peak_idx - context_window)
    hi = min(T - 1, peak_idx + context_window)
    return [Image.fromarray(frames_np[i]) for i in range(lo, hi + 1)]

def extract_all_views_flow(hdf5_file, action_key: str, clips: list, max_views: int = 4) -> List[List[Image.Image]]:
    frames_per_view = []
    for clip_raw in clips[:max_views]:
        clip_key = clip_raw.replace(".mp4", "")
        frames = extract_flow_keyframe(hdf5_file, action_key, clip_key)
        if frames: frames_per_view.append(frames)
    return frames_per_view
