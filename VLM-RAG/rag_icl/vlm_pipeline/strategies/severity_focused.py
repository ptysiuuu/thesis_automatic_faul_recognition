"""
strategies/severity_focused.py
===============================
New ablation strategies targeting the severity bottleneck.
All use Qwen2.5-VL-7B — the reliable baseline model.

Current problem: cos_two_stage gets 51.09 Act BA but only 27.34 Sev BA.
Root cause: key-frame selection (contact moment) loses approach/aftermath
            which encode force information critical for severity judgment.

Strategies here:
  Row 5 — full_sev_two_stage:
    CoS for action (key frames), but ALL frames for severity.
    Hypothesis: action needs contact frame, severity needs full temporal context.

  Row 6 — per_action_prior:
    static_few_shot + per-action severity priors in prompt.
    Hypothesis: telling the model "Elbowing is usually Red card" prevents
    defaulting to Yellow card for everything.

  Row 7 — targeted_retrieval:
    Two-stage with targeted medoid examples (same action + confusable boundary).
    Fixes data_driven's collapse: show fewer, more relevant examples.

  Row 8 — ordinal_severity:
    Reframe severity as ordinal step-by-step decision (0→3).
    Hypothesis: structured reasoning about force levels improves calibration.

  Row 9 — cos_full_sev (best combined):
    CoS frame selection for action + full frames + ordinal severity + per-action priors.
    Best of everything in one system.
"""

from typing import List, Tuple, Dict
from PIL import Image

from .base import parse_response, parse_action_only, parse_severity_only
from ..utils.constants import (
    ACTION_CLASSES,
    SEVERITY_CLASSES,
    PER_ACTION_SEVERITY_PRIOR,
)
from ..utils.frames import (
    parse_key_frames,
    select_key_frames,
    format_selected_frame_info,
)
from ..prompts.builders import (
    build_cos_frame_selection_prompt,
    build_cos_action_disambig_prompt,
    build_cos_static_severity_prompt,
    build_description_prompt,
    build_severity_from_description_prompt,
    build_cos_action_prompt,
    build_full_frame_severity_prompt,
    build_static_prompt,
    build_per_action_prior_prompt,
    build_two_stage_action_prompt,
    build_two_stage_severity_prompt,
    build_targeted_retrieval_prompt,
    build_ordinal_severity_prompt,
    build_cos_severity_prompt,
)
from ..retrieval.medoid_cache import (
    build_examples_text,
    build_targeted_examples,
    build_severity_examples,
)
from ..utils.constants import SEVERITY_TO_IDX
from ..prompts.templates import SEVERITY_LIST_STR
from ..prompts.builders import (
    build_contrastive_severity_prompt,
    build_physics_flow_prompt,
)
import base64
import io
import numpy as _np

try:
    import cv2
except Exception:
    cv2 = None

# ── Row 5: full_sev_two_stage ─────────────────────────────────────────────────


def run_full_sev_two_stage(
    backend,
    frames_per_view: List[List[Image.Image]],
    law12_ctx: str,
    medoid_cache: dict,
    severity_priors: dict,
    per_action_priors: dict,
    frames_per_view_count: int,
    action_hint: str,
    rag,
) -> Tuple[int, int, str]:
    """
    Stage 0: CoS frame selection (Qwen picks contact frame)
    Stage 1: Action classification using key frames
    Stage 2: Severity classification using ALL frames (not key frames)

    The key insight: severity needs approach + contact + aftermath.
    Key frames are great for identifying WHAT happened, not HOW HARD.
    """
    n_views = len(frames_per_view)

    # Stage 0: frame selection
    sel_prompt = build_cos_frame_selection_prompt(n_views, frames_per_view_count)
    raw0 = backend.classify(frames_per_view, sel_prompt)
    key_indices = parse_key_frames(raw0, n_views, frames_per_view_count)
    key_frames = select_key_frames(frames_per_view, key_indices, context_window=1)
    sel_info = format_selected_frame_info(key_indices, n_views)

    # Stage 1: action from key frames
    all_text, all_imgs = build_examples_text(medoid_cache, n_per_class=1)
    act_prompt = build_cos_action_prompt(
        n_views=n_views,
        law12_context=law12_ctx,
        mined_examples=all_text,
        selected_frame_info=sel_info,
    )
    raw1 = backend.classify(key_frames, act_prompt, extra_images=all_imgs)
    act_idx = parse_action_only(raw1)
    act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

    # Stage 2: severity from ALL frames (full temporal context)
    sev_examples_text, sev_imgs = build_severity_examples(medoid_cache, act_str)
    sev_law12 = rag.retrieve(rag.build_query(act_str))
    per_action = per_action_priors.get(act_str, severity_priors)
    sev_prompt = build_full_frame_severity_prompt(
        n_views=n_views,
        law12_context=sev_law12,
        predicted_action=act_str,
        severity_examples=sev_examples_text,
        severity_priors=per_action,
    )

    # KEY: use FULL frames for severity, not key_frames
    raw2 = backend.classify(frames_per_view, sev_prompt, extra_images=sev_imgs)
    sev_idx = parse_severity_only(raw2)

    raw = (
        f"STAGE0 (key frames): {raw0}\n"
        f"Selected: {sel_info}\n"
        f"STAGE1 (action, key frames): {raw1}\n"
        f"STAGE2 (severity, FULL frames): {raw2}"
    )
    return act_idx, sev_idx, raw


# ── Row 6: per_action_prior ───────────────────────────────────────────────────


def run_per_action_prior(
    backend,
    frames_per_view: List[List[Image.Image]],
    law12_ctx: str,
    per_action_priors: dict,
) -> Tuple[int, int, str]:
    """
    static_few_shot with per-action severity calibration added to the prompt.
    Single VLM call — same speed as static_few_shot.
    """
    prompt = build_per_action_prior_prompt(
        n_views=len(frames_per_view),
        law12_context=law12_ctx,
        per_action_priors=per_action_priors,
    )
    raw = backend.classify(frames_per_view, prompt)
    act_idx, sev_idx = parse_response(raw)
    return act_idx, sev_idx, raw


# ── Row 7: targeted_retrieval ─────────────────────────────────────────────────


def run_targeted_retrieval(
    backend,
    frames_per_view: List[List[Image.Image]],
    law12_ctx: str,
    medoid_cache: dict,
    severity_priors: dict,
    per_action_priors: dict,
    rag,
) -> Tuple[int, int, str]:
    """
    Two-stage with targeted examples:
    Stage 1: action classification (all frames, targeted medoid examples)
    Stage 2: severity classification (all frames, same-action + boundary examples)

    Fixes data_driven by only showing relevant examples for the predicted action.
    """
    n_views = len(frames_per_view)

    # Stage 1: action with all medoid examples (needed since we don't know action yet)
    all_text, all_imgs = build_examples_text(medoid_cache, n_per_class=1)
    act_prompt = build_two_stage_action_prompt(
        n_views=n_views, law12_context=law12_ctx, mined_examples=all_text
    )
    raw1 = backend.classify(frames_per_view, act_prompt, extra_images=all_imgs)
    act_idx = parse_action_only(raw1)
    act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

    # Stage 2: severity with TARGETED examples (same action + confusable boundary)
    targeted_text, targeted_imgs = build_targeted_examples(
        medoid_cache, predicted_action=act_str, n_same_action=2, n_confusable=1
    )
    sev_law12 = rag.retrieve(rag.build_query(act_str))
    per_action = per_action_priors.get(act_str, severity_priors)

    sev_prompt = build_targeted_retrieval_prompt(
        n_views=n_views,
        law12_context=sev_law12,
        targeted_examples=targeted_text,
        predicted_action_hint=act_str,
    )
    raw2 = backend.classify(frames_per_view, sev_prompt, extra_images=targeted_imgs)
    sev_idx = parse_severity_only(raw2)

    raw = f"STAGE1 (action): {raw1}\nSTAGE2 (targeted severity): {raw2}"
    return act_idx, sev_idx, raw


# ── Row 8: ordinal_severity ───────────────────────────────────────────────────


def run_ordinal_severity(
    backend,
    frames_per_view: List[List[Image.Image]],
    law12_ctx: str,
    medoid_cache: dict,
    severity_priors: dict,
    per_action_priors: dict,
    rag,
) -> Tuple[int, int, str]:
    """
    Two-stage where severity uses ordinal step-by-step reasoning (0→3).
    Forces model through: contact? → ball-aimed? → reckless? → excessive?
    Each step narrows the severity level, preventing Yellow card collapse.
    """
    n_views = len(frames_per_view)

    # Stage 1: action (standard two-stage action prompt)
    all_text, all_imgs = build_examples_text(medoid_cache, n_per_class=1)
    act_prompt = build_two_stage_action_prompt(
        n_views=n_views, law12_context=law12_ctx, mined_examples=all_text
    )
    raw1 = backend.classify(frames_per_view, act_prompt, extra_images=all_imgs)
    act_idx = parse_action_only(raw1)
    act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

    # Stage 2: ordinal severity reasoning
    sev_examples_text, sev_imgs = build_severity_examples(medoid_cache, act_str)
    sev_law12 = rag.retrieve(rag.build_query(act_str))
    per_action = per_action_priors.get(act_str, severity_priors)

    sev_prompt = build_ordinal_severity_prompt(
        n_views=n_views,
        law12_context=sev_law12,
        predicted_action=act_str,
        severity_examples=sev_examples_text,
        severity_priors=per_action,
    )
    raw2 = backend.classify(frames_per_view, sev_prompt, extra_images=sev_imgs)
    sev_idx = parse_severity_only(raw2)

    raw = f"STAGE1 (action): {raw1}\nSTAGE2 (ordinal severity): {raw2}"
    return act_idx, sev_idx, raw


# ── Row 9: cos_full_sev (best combined) ──────────────────────────────────────


def run_cos_full_sev(
    backend,
    frames_per_view: List[List[Image.Image]],
    law12_ctx: str,
    medoid_cache: dict,
    severity_priors: dict,
    per_action_priors: dict,
    frames_per_view_count: int,
    rag,
) -> Tuple[int, int, str]:
    """
    Best combined system:
    Stage 0: CoS frame selection
    Stage 1: Action from key frames + targeted examples
    Stage 2: Severity from ALL frames + ordinal reasoning + per-action priors

    Combines:
    - CoS frame selection (best action improvement)
    - Full frames for severity (best severity improvement)
    - Ordinal reasoning (structured force assessment)
    - Per-action priors (calibration)
    - Targeted examples (relevant context without confusion)
    """
    n_views = len(frames_per_view)

    # Stage 0: CoS frame selection
    sel_prompt = build_cos_frame_selection_prompt(n_views, frames_per_view_count)
    raw0 = backend.classify(frames_per_view, sel_prompt)
    key_indices = parse_key_frames(raw0, n_views, frames_per_view_count)
    key_frames = select_key_frames(frames_per_view, key_indices, context_window=1)
    sel_info = format_selected_frame_info(key_indices, n_views)

    # Stage 1: action from key frames with targeted examples
    all_text, all_imgs = build_examples_text(medoid_cache, n_per_class=1)
    act_prompt = build_cos_action_disambig_prompt(
        n_views=n_views,
        law12_context=law12_ctx,
        mined_examples=all_text,
        selected_frame_info=sel_info,
    )
    raw1 = backend.classify(key_frames, act_prompt, extra_images=all_imgs)
    act_idx = parse_action_only(raw1)
    act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

    # Stage 2: ordinal severity from ALL frames + per-action priors
    sev_examples_text, sev_imgs = build_severity_examples(medoid_cache, act_str)
    sev_law12 = rag.retrieve(rag.build_query(act_str))
    per_action = per_action_priors.get(act_str, severity_priors)

    sev_prompt = build_ordinal_severity_prompt(
        n_views=n_views,
        law12_context=sev_law12,
        predicted_action=act_str,
        severity_examples=sev_examples_text,
        severity_priors=per_action,
    )

    # KEY: full frames for severity
    raw2 = backend.classify(frames_per_view, sev_prompt, extra_images=sev_imgs)
    sev_idx = parse_severity_only(raw2)

    # ── Poprawiona końcówka funkcji run_cos_full_sev ──────────────────────────────
    raw = (
        f"STAGE0 (CoS selection): {raw0}\n"
        f"Selected: {sel_info}\n"
        f"STAGE1 (action, key frames): {raw1}\n"
        f"STAGE2 (ordinal severity, full frames): {raw2}"
    )
    return act_idx, sev_idx, raw


# ── Poprawiona nowa funkcja ───────────────────────────────────────────────────
def run_cos_two_stage_description_severity(
    backend,
    frames_per_view: List[List[Image.Image]],
    law12_ctx: str,
    medoid_cache: dict,
    frames_per_view_count: int,
    rag,
) -> Tuple[int, int, str]:
    """
    Keep CoS action classification identical, but replace severity with
    a description-first text reasoning stage. Steps:
      1) CoS frame selection -> key frames
      2) Action classification from key frames (with medoid examples)
      3) Description prompt on 3 key frames per view (approach/contact/aftermath)
      4) Severity inference from description + action (text only)
    """
    n_views = len(frames_per_view)

    # Stage 0: CoS frame selection
    sel_prompt = build_cos_frame_selection_prompt(n_views, frames_per_view_count)
    raw0 = backend.classify(frames_per_view, sel_prompt)
    key_indices = parse_key_frames(raw0, n_views, frames_per_view_count)
    key_frames = select_key_frames(frames_per_view, key_indices, context_window=1)
    sel_info = format_selected_frame_info(key_indices, n_views)

    # Stage 1: action from key frames (same as cos_two_stage)
    all_text, all_imgs = build_examples_text(medoid_cache, n_per_class=1)
    act_prompt = build_cos_action_disambig_prompt(
        n_views=n_views,
        law12_context=law12_ctx,
        mined_examples=all_text,
        selected_frame_info=sel_info,
    )
    raw1 = backend.classify(key_frames, act_prompt, extra_images=all_imgs)
    act_idx = parse_action_only(raw1)
    act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

    # Stage 2: description from 3 SELECTED KEY frames per view
    def _three_frames_per_view(fpv):
        out = []
        for frames in fpv:
            if not frames:
                continue
            idxs = [0, len(frames) // 2, len(frames) - 1]
            seen = []
            for i in idxs:
                i = max(0, min(i, len(frames) - 1))
                if i not in seen:
                    seen.append(i)
            out.append([frames[i] for i in seen])
        return out

    # ZMIANA: Przekazujemy key_frames zamiast frames_per_view!
    desc_frames = _three_frames_per_view(key_frames)

    desc_prompt = build_description_prompt(desc_frames, law12_ctx)
    raw_desc = backend.classify(desc_frames, desc_prompt)
    description = raw_desc.strip()

    # Stage 3: severity from description + action (text-only)
    sev_prompt = build_severity_from_description_prompt(act_str, description, law12_ctx)
    raw2 = backend.classify([], sev_prompt)
    sev_idx = parse_severity_only(raw2)

    raw = (
        f"STAGE0 (CoS selection): {raw0}\nSelected: {sel_info}\n"
        f"STAGE1 (action): {raw1}\nSTAGE2 (description): {raw_desc}\nSTAGE3 (severity): {raw2}"
    )
    return act_idx, sev_idx, raw


# ── New: Contrastive severity anchoring (Architecture 2) ───────────────────
def run_contrastive_severity(
    backend,
    frames_per_view: List[List[Image.Image]],
    law12_ctx: str,
    medoid_cache: dict,
    severity_priors: dict,
    per_action_priors: dict,
    rag,
) -> Tuple[int, int, str]:
    """
    Architecture 2 — Contrastive Severity Anchoring

    Stage 1: predict action (two-stage action prompt)
    Stage 2: bracketed binary comparisons against medoid anchors for the
             predicted action. Map binary tournament winner to final severity.
    """
    n_views = len(frames_per_view)

    # Stage 1: action classification (reuse two-stage action prompt)
    all_text, all_imgs = build_examples_text(medoid_cache, n_per_class=1)
    act_prompt = build_two_stage_action_prompt(
        n_views=n_views, law12_context=law12_ctx, mined_examples=all_text
    )
    raw1 = backend.classify(frames_per_view, act_prompt, extra_images=all_imgs)
    act_idx = parse_action_only(raw1)
    act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

    # Gather anchors for this action from medoid_cache
    anchors = {}
    for key, entry in medoid_cache.items():
        if entry.get("action") != act_str:
            continue
        anchors[entry.get("severity")] = entry

    # Severity ladder
    ladder = ["No offence", "No card", "Yellow card", "Red card"]

    def _get_anchor_image(entry):
        from PIL import Image as PILImage

        if not entry:
            return None
        b64 = entry["frames_b64"][0]
        return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    # Bracket tournament
    winner = "No offence"
    for challenger in ["No card", "Yellow card", "Red card"]:
        anchor_a = _get_anchor_image(anchors.get(winner))
        anchor_b = _get_anchor_image(anchors.get(challenger))

        # If either anchor is missing, skip this challenger
        if anchor_a is None or anchor_b is None:
            # if both missing, skip; if only challenger missing, keep winner
            continue

        # Build prompt: ask which anchor the TEST CLIP is closer to
        prompt = build_contrastive_severity_prompt(
            n_views=n_views,
            law12_context=law12_ctx,
            action=act_str,
            anchor_a_sev=winner,
            anchor_b_sev=challenger,
        )

        # Order of extra_images: anchor_a, anchor_b
        raw = backend.classify(
            frames_per_view, prompt, extra_images=[anchor_a, anchor_b]
        )

        # parse simple 'choice' number from response (robust fallback)
        choice = None
        try:
            import json as _json

            j = _json.loads(raw)
            choice = int(j.get("choice", None))
        except Exception:
            txt = raw.strip().lower()
            if txt.startswith("1"):
                choice = 1
            elif txt.startswith("2"):
                choice = 2

        if choice == 2:
            winner = challenger

    final_sev = winner
    sev_idx = SEVERITY_TO_IDX.get(final_sev, -1)
    raw_summary = f"ACTION: {raw1}\nCONTRASTIVE_TOURNAMENT_WINNER: {final_sev}"
    return act_idx, sev_idx, raw_summary


# ── New: Physics-grounded severity (Architecture 3) ────────────────────────
def run_physics_severity(
    backend,
    frames_per_view: List[List[Image.Image]],
    law12_ctx: str,
    medoid_cache: dict,
    severity_priors: dict,
    per_action_priors: dict,
    rag,
) -> Tuple[int, int, str]:
    """
    Architecture 3 — Physics-grounded severity using optical flow statistics.

    Lightweight implementation: compute dense flow between contact frame and
    aftermath frames, take max displacement as proxy, and map via heuristic
    thresholds. If OpenCV is not available, fallback to VLM severity.
    """
    n_views = len(frames_per_view)

    # Stage 1: action classification (re-use two-stage action)
    all_text, all_imgs = build_examples_text(medoid_cache, n_per_class=1)
    act_prompt = build_two_stage_action_prompt(
        n_views=n_views, law12_context=law12_ctx, mined_examples=all_text
    )
    raw1 = backend.classify(frames_per_view, act_prompt, extra_images=all_imgs)
    act_idx = parse_action_only(raw1)
    act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

    # Select contact frame via CoS selection
    sel_prompt = build_cos_frame_selection_prompt(
        n_views,
        frames_per_view_count=(
            len(frames_per_view[0]) if frames_per_view and frames_per_view[0] else 1
        ),
    )
    raw0 = backend.classify(frames_per_view, sel_prompt)
    key_idx = parse_key_frames(
        raw0,
        n_views,
        len(frames_per_view[0]) if frames_per_view and frames_per_view[0] else 1,
    )

    # Compute flow stats across views (if cv2 available)
    max_disp = 0.0
    if cv2 is None:
        # Fallback: ask VLM for severity using full-frame severity prompt
        sev_examples_text, sev_imgs = build_severity_examples(medoid_cache, act_str)
        sev_law12 = rag.retrieve(rag.build_query(act_str))
        per_action = per_action_priors.get(act_str, severity_priors)
        sev_prompt = build_full_frame_severity_prompt(
            n_views=n_views,
            law12_context=sev_law12,
            predicted_action=act_str,
            severity_examples=sev_examples_text,
            severity_priors=per_action,
        )
        raw2 = backend.classify(frames_per_view, sev_prompt, extra_images=sev_imgs)
        sev_idx = parse_severity_only(raw2)
        return act_idx, sev_idx, f"Fallback VLM severity: {raw2}"

    # Otherwise, compute optical flow between contact frame and last frame per view
    for v_idx, frames in enumerate(frames_per_view):
        if not frames:
            continue
        contact_i = key_idx[v_idx] if v_idx < len(key_idx) else 0
        contact_i = max(0, min(contact_i, len(frames) - 1))
        img0 = _np.array(frames[contact_i].convert("RGB"))
        img1 = _np.array(frames[-1].convert("RGB"))
        gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        max_disp = max(max_disp, float(mag.max()))

    # Map displacement to severity via heuristic thresholds (to be trained later)
    if max_disp < 2.0:
        final_sev = "No offence"
    elif max_disp < 6.0:
        final_sev = "No card"
    elif max_disp < 12.0:
        final_sev = "Yellow card"
    else:
        final_sev = "Red card"

    sev_idx = SEVERITY_TO_IDX.get(final_sev, -1)
    # Optionally ask VLM to render a human-readable justification using the flow stat
    physics_prompt = build_physics_flow_prompt(
        action=act_str, max_disp=max_disp, severity_list=SEVERITY_LIST_STR
    )
    raw_vlm = backend.classify([], physics_prompt)
    raw = f"FLOW_MAX_DISP: {max_disp:.3f} — VLM_EXPLAIN: {raw_vlm}"
    return act_idx, sev_idx, raw
    return act_idx, sev_idx, raw


def run_cos_static_sev(
    backend,
    frames_per_view: List[List[Image.Image]],
    law12_ctx: str,
    medoid_cache: dict,
    severity_priors: dict,
    per_action_priors: dict,
    frames_per_view_count: int,
    rag,
) -> Tuple[int, int, str]:
    """
    CoS + Static severity variant:
    Stage 0: CoS frame selection (key frames for contact moment)
    Stage 1: Action from key frames + medoid examples
    Stage 2: Severity from ALL frames with STATIC_EXAMPLES (no medoid retrieval)

    Hypothesis: static text examples (manually curated) work better than
    medoid retrieval for severity calibration, while keeping full frames
    to preserve force/approach information.
    """
    n_views = len(frames_per_view)

    # Stage 0: CoS frame selection
    sel_prompt = build_cos_frame_selection_prompt(n_views, frames_per_view_count)
    raw0 = backend.classify(frames_per_view, sel_prompt)
    key_indices = parse_key_frames(raw0, n_views, frames_per_view_count)
    key_frames = select_key_frames(frames_per_view, key_indices, context_window=1)
    sel_info = format_selected_frame_info(key_indices, n_views)

    # Stage 1: action from key frames with medoid examples
    all_text, all_imgs = build_examples_text(medoid_cache, n_per_class=1)
    act_prompt = build_cos_action_disambig_prompt(
        n_views=n_views,
        law12_context=law12_ctx,
        mined_examples=all_text,
        selected_frame_info=sel_info,
    )
    raw1 = backend.classify(key_frames, act_prompt, extra_images=all_imgs)
    act_idx = parse_action_only(raw1)
    act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

    # Stage 2: severity from ALL frames with static examples (no medoid)
    per_action = per_action_priors.get(act_str, severity_priors)
    sev_prompt = build_cos_static_severity_prompt(
        n_views=n_views,
        law12_context=law12_ctx,
        predicted_action=act_str,
        severity_priors=per_action,
    )

    # KEY: use FULL frames for severity, with static examples
    raw2 = backend.classify(frames_per_view, sev_prompt)
    sev_idx = parse_severity_only(raw2)

    raw = (
        f"STAGE0 (CoS selection): {raw0}\n"
        f"Selected: {sel_info}\n"
        f"STAGE1 (action, key frames): {raw1}\n"
        f"STAGE2 (static severity, full frames): {raw2}"
    )
    return act_idx, sev_idx, raw


def run_flow_hard_neg(
    backend,
    frames_per_view: List[List[Image.Image]],
    law12_ctx: str,
    medoid_cache: dict,
    severity_priors: dict,
    rag,
    retriever,
) -> Tuple[int, int, str]:
    """Stage 1: Action (Flow frames). Stage 2: Severity (Flow frames + Hard Negatives)."""
    n_views = len(frames_per_view)

    # STAGE 1: Action (using OpenCV flow frames, no CoS VLM call needed)
    all_text, all_imgs = build_examples_text(medoid_cache, n_per_class=1)
    act_prompt = build_two_stage_action_prompt(n_views, law12_ctx, all_text)
    raw1 = backend.classify(frames_per_view, act_prompt, extra_images=all_imgs)
    act_idx = parse_action_only(raw1)
    act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

    # STAGE 2: Severity with Hard Negatives
    hard_neg_text = retriever.retrieve_hard_negatives(frames_per_view[0], act_str)
    sev_law12 = rag.retrieve(rag.build_query(act_str))

    # Reusing the targeted retrieval prompt template for hard negatives
    sev_prompt = build_targeted_retrieval_prompt(
        n_views=n_views,
        law12_context=sev_law12,
        targeted_examples=hard_neg_text,
        predicted_action_hint=act_str,
    )
    raw2 = backend.classify(frames_per_view, sev_prompt)
    sev_idx = parse_severity_only(raw2)

    return (
        act_idx,
        sev_idx,
        f"STAGE1 (Flow Action): {raw1}\nSTAGE2 (Hard Neg Severity): {raw2}",
    )
