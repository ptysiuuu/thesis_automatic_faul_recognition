"""
strategies/multi_turn_var.py
============================
Multi-Turn VAR: 3-turn multi-view foul classification strategy.

Turn 1 — Physical description only (5 frames: 2 live + 3 replay):
  One live approach frame + one live contact frame establish spatial context.
  One replay approach + one replay CONTACT FRAME + one replay aftermath provide
  the primary evidence. Model produces a structured visual description — no
  classification yet.  Separating perception from rule-application is the key
  fix for the 42.9% tool-grounding failure observed in cos_two_stage.

Turn 2 — Forced-coverage action classification (4 dense contact frames):
  All 4 frames come from the closest replay, tightly clustered around contact.
  The forced-coverage checklist requires an explicit Yes/No/Uncertain verdict
  for every one of the 8 classes before a final answer is committed, preventing
  the silent collapse onto the 5 dominant classes.
  Turn 1 description is echoed verbatim so the model cannot contradict its own
  earlier physical evidence.

Turn 3 — Ordinal severity cascade (4 aftermath-biased frames):
  Contact frame + three aftermath frames (0.5 / 1.0 / 1.5s after contact).
  If a second replay is available, the 1.5s frame comes from a perpendicular
  angle to improve fall-distance estimation.
  Severity is decided via an explicit 4-step IFAB cascade that mirrors how a
  referee actually escalates from careless → reckless → excessive.
  Both the predicted action type and the Turn 1 description are echoed to
  maintain the evidential chain and prevent contradictions.
"""

from typing import List, Tuple
from PIL import Image

from .base import parse_action_only, parse_severity_only
from ..utils.constants import ACTION_CLASSES
from ..prompts.builders import (
    build_multi_turn_turn1_prompt,
    build_multi_turn_turn2_prompt,
    build_multi_turn_turn3_prompt,
)


# ── Temporal helpers ──────────────────────────────────────────────────────────

def _temporal_bin(proportion: float) -> str:
    """Map clip proportion (0–1) to a named temporal zone."""
    if proportion < 0.25:
        return "far-before"
    if proportion < 0.45:
        return "approach"
    if proportion < 0.60:
        return "contact-window"
    if proportion < 0.80:
        return "immediate-aftermath"
    return "settled"


def _safe_frame(frames: List[Image.Image], proportion: float) -> Tuple[Image.Image, int]:
    """Return (frame, absolute_index) at a given clip proportion with clamping."""
    n = len(frames)
    idx = min(int(proportion * n), n - 1)
    idx = max(0, idx)
    return frames[idx], idx


def _dedup_indices(raw: List[int], n_total: int, want: int) -> List[int]:
    """
    Ensure `want` distinct indices from raw, falling back to even spacing if
    the extracted sequence is too short to yield distinct proportional indices.
    """
    seen: set = set()
    unique = []
    for i in raw:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    if len(unique) >= want:
        return unique[:want]
    # Fallback: evenly-spaced indices centred on clip
    import numpy as np
    center = n_total // 2
    spread = max(want - 1, 1)
    start = max(0, center - spread // 2)
    end = min(n_total - 1, start + spread)
    return list(np.linspace(start, end, min(want, n_total), dtype=int))


# ── Frame selection for each turn ─────────────────────────────────────────────

def _select_turn1_frames(
    fpv: List[List[Image.Image]],
) -> Tuple[List[Image.Image], List[str]]:
    """
    5 frames: 2 live camera + 3 closest-replay.

    Live frames establish field position and approach vector; replay frames
    provide the primary close-up evidence, with one frame explicitly marked as
    the CONTACT FRAME (mid-clip anchor).
    """
    live = fpv[0] if len(fpv) >= 1 else []
    rep1 = fpv[1] if len(fpv) >= 2 else (fpv[0] if fpv else [])

    frames: List[Image.Image] = []
    labels: List[str] = []

    # 2 live frames: approach (~25 %) and contact (~50 %)
    for p, desc in [(0.25, "approach — ~2 s before contact"),
                    (0.50, "contact-window")]:
        if live:
            f, fi = _safe_frame(live, p)
            frames.append(f)
            labels.append(
                f"[Live Camera | Frame {fi} | {_temporal_bin(p)}] — {desc}"
            )

    # 3 replay frames: 1 s before, CONTACT FRAME, 1 s after
    specs = [
        (0.35, "approach — ~1 s before contact", False),
        (0.50, "contact-window",                 True),   # CONTACT FRAME
        (0.65, "immediate-aftermath — ~1 s after contact", False),
    ]
    for p, desc, is_contact in specs:
        if rep1:
            f, fi = _safe_frame(rep1, p)
            frames.append(f)
            marker = " ← CONTACT FRAME" if is_contact else ""
            labels.append(
                f"[Replay 1 | Frame {fi} | {_temporal_bin(p)}]{marker} — {desc}"
            )

    return frames, labels


def _select_turn2_frames(
    fpv: List[List[Image.Image]],
) -> Tuple[List[Image.Image], List[str]]:
    """
    4 dense frames from the closest replay, clustered tightly around contact.
    Proportions: 30 % / 43 % / 50 % (CONTACT) / 60 %.
    Falls back to even centre-spread when the clip is too short for distinct
    proportional indices.
    """
    rep1 = fpv[1] if len(fpv) >= 2 else (fpv[0] if fpv else [])
    if not rep1:
        return [], []

    n = len(rep1)
    target_proportions = [0.30, 0.43, 0.50, 0.60]
    contact_slot = 2  # the 0.50 proportion entry is the CONTACT FRAME

    raw_indices = [min(int(p * n), n - 1) for p in target_proportions]
    indices = _dedup_indices(raw_indices, n, 4)

    # Re-identify which index is closest to the 50 % contact point
    if len(indices) >= 3:
        contact_slot = min(range(len(indices)), key=lambda i: abs(indices[i] - n // 2))

    frames: List[Image.Image] = []
    labels: List[str] = []
    for i, fi in enumerate(indices):
        p = fi / max(n - 1, 1)
        marker = " ← CONTACT FRAME" if i == contact_slot else ""
        frames.append(rep1[fi])
        labels.append(f"[Replay 1 | Frame {fi} | {_temporal_bin(p)}]{marker}")

    return frames, labels


def _select_turn3_frames(
    fpv: List[List[Image.Image]],
) -> Tuple[List[Image.Image], List[str]]:
    """
    4 aftermath-biased frames: CONTACT FRAME + 0.5 s / 1.0 s / 1.5 s after.
    The 1.5 s frame uses Replay 2 if available (perpendicular angle aids
    fall-distance and injury assessment).
    """
    rep1 = fpv[1] if len(fpv) >= 2 else (fpv[0] if fpv else [])
    rep2 = fpv[2] if len(fpv) >= 3 else None
    if not rep1:
        return [], []

    n1 = len(rep1)
    frames: List[Image.Image] = []
    labels: List[str] = []

    # Contact frame
    f, fi = _safe_frame(rep1, 0.50)
    frames.append(f)
    labels.append(f"[Replay 1 | Frame {fi} | contact-window] ← CONTACT FRAME")

    # 0.5 s and 1.0 s after
    for p, tag in [(0.62, "0.5 s after contact"), (0.74, "1.0 s after contact")]:
        f, fi = _safe_frame(rep1, p)
        frames.append(f)
        labels.append(f"[Replay 1 | Frame {fi} | immediate-aftermath] ({tag})")

    # 1.5 s after — prefer second replay for perpendicular angle
    if rep2 and len(rep2) > 0:
        f, fi = _safe_frame(rep2, 0.74)
        frames.append(f)
        labels.append(
            f"[Replay 2 | Frame {fi} | immediate-aftermath] (1.5 s after, perpendicular view)"
        )
    else:
        f, fi = _safe_frame(rep1, 0.86)
        frames.append(f)
        labels.append(f"[Replay 1 | Frame {fi} | settled] (1.5 s after contact)")

    return frames, labels


# ── Main strategy function ────────────────────────────────────────────────────

def run_multi_turn_var(
    backend,
    frames_per_view: List[List[Image.Image]],
    law12_ctx: str,
    per_action_priors: dict,
    rag,
) -> Tuple[int, int, str]:
    """
    Three-turn multi-view VAR strategy.

    Parameters
    ----------
    backend          : Any VLM backend (Qwen, NVIDIA, Gemini, …)
    frames_per_view  : Extracted frames, one inner list per camera view.
                       Weighted 8-frame extraction recommended; 4-frame works.
    law12_ctx        : Law 12 RAG context (general query, used in Turn 1 & 2).
    per_action_priors: {action_name: {severity_name: pct}} calibration dict.
    rag              : Law12RAG instance (used in Turn 3 for action-specific query).

    Returns
    -------
    (act_idx, sev_idx, raw_combined_text)
    """
    if not frames_per_view:
        return -1, -1, "No frames available"

    # ── Turn 1: physical description ─────────────────────────────────────────
    t1_frames, t1_labels = _select_turn1_frames(frames_per_view)
    if not t1_frames:
        return -1, -1, "Turn 1 frame selection failed (no views)"

    t1_prompt = build_multi_turn_turn1_prompt(t1_labels, law12_ctx)
    # Pass frames as extra_images so the backend does not inject redundant
    # "Live camera / Replay N" view-labels that would conflict with our labels.
    raw1 = backend.classify([], t1_prompt, extra_images=t1_frames)
    description = raw1.strip()

    # ── Turn 2: forced-coverage action classification ─────────────────────────
    t2_frames, t2_labels = _select_turn2_frames(frames_per_view)
    if not t2_frames:
        # Graceful fallback to a subset of Turn 1 frames
        t2_frames = t1_frames[:4]
        t2_labels = t1_labels[:4]

    t2_prompt = build_multi_turn_turn2_prompt(t2_labels, description)
    raw2 = backend.classify([], t2_prompt, extra_images=t2_frames)
    act_idx = parse_action_only(raw2)
    act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

    # ── Turn 3: ordinal severity cascade ──────────────────────────────────────
    t3_frames, t3_labels = _select_turn3_frames(frames_per_view)
    if not t3_frames:
        t3_frames = t1_frames
        t3_labels = t1_labels

    # Use action-specific Law 12 context for severity turn
    sev_law12 = rag.retrieve(rag.build_query(act_str))
    per_action = per_action_priors.get(act_str, {})

    t3_prompt = build_multi_turn_turn3_prompt(
        frame_labels=t3_labels,
        action_type=act_str,
        turn1_description=description,
        law12_ctx=sev_law12,
        per_action_priors=per_action,
    )
    raw3 = backend.classify([], t3_prompt, extra_images=t3_frames)
    sev_idx = parse_severity_only(raw3)

    raw = (
        f"TURN1 (description):\n{raw1}\n\n"
        f"TURN2 (action [{act_str}]):\n{raw2}\n\n"
        f"TURN3 (severity):\n{raw3}"
    )
    return act_idx, sev_idx, raw
