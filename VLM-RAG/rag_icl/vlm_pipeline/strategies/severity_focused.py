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

    raw = (
        f"STAGE0 (CoS selection): {raw0}\n"
        f"Selected: {sel_info}\n"
        f"STAGE1 (action, key frames): {raw1}\n"
        f"STAGE2 (ordinal severity, full frames): {raw2}"
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
