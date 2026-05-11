"""
prompts/builders.py
===================
Functions that fill prompt templates with runtime values.
Each builder takes structured arguments and returns a prompt string.
"""

from typing import List, Dict
from .templates import (
    ACTION_LIST_STR,
    SEVERITY_LIST_STR,
    STATIC_EXAMPLES,
    STATIC_FEW_SHOT_TMPL,
    DATA_DRIVEN_TMPL,
    TWO_STAGE_ACTION_TMPL,
    TWO_STAGE_SEVERITY_TMPL,
    DESCRIPTION_FIRST_DESCRIPTION_TMPL,
    DESCRIPTION_FIRST_SEVERITY_TMPL,
    RAGICL_TMPL,
    COS_FRAME_SELECT_TMPL,
    COS_ACTION_TMPL,
    COS_SEVERITY_TMPL,
    FULL_FRAME_SEVERITY_TMPL,
    PER_ACTION_PRIOR_TMPL,
    TARGETED_RETRIEVAL_TMPL,
    ORDINAL_SEVERITY_TMPL,
    CONTRASTIVE_SEVERITY_TMPL,
    PHYSICS_FLOW_TMPL,
)
from ..utils.constants import SEVERITY_CLASSES


def _prior_str(priors: dict) -> str:
    return ", ".join(f"{k}: {v}%" for k, v in priors.items())


def _per_action_prior_str(per_action_priors: dict) -> str:
    lines = []
    for action, priors in sorted(per_action_priors.items()):
        lines.append(f"  {action}: {_prior_str(priors)}")
    return "\n".join(lines)


# ── Row 0 ──────────────────────────────────────────────────────────────────────
def build_static_prompt(n_views: int, law12_context: str) -> str:
    return STATIC_FEW_SHOT_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        examples=STATIC_EXAMPLES,
        action_list=ACTION_LIST_STR,
        severity_list=SEVERITY_LIST_STR,
    )


# ── Row 1 ──────────────────────────────────────────────────────────────────────
def build_data_driven_prompt(
    n_views: int, law12_context: str, mined_examples: str, severity_priors: dict
) -> str:
    return DATA_DRIVEN_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        mined_examples=mined_examples,
        prior_str=_prior_str(severity_priors),
        action_list=ACTION_LIST_STR,
        severity_list=SEVERITY_LIST_STR,
    )


# ── Row 2 ──────────────────────────────────────────────────────────────────────
def build_two_stage_action_prompt(
    n_views: int, law12_context: str, mined_examples: str
) -> str:
    return TWO_STAGE_ACTION_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        mined_examples=mined_examples,
        action_list=ACTION_LIST_STR,
    )


def build_two_stage_severity_prompt(
    n_views: int,
    law12_context: str,
    predicted_action: str,
    severity_examples: str,
    severity_priors: dict,
) -> str:
    return TWO_STAGE_SEVERITY_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        predicted_action=predicted_action,
        severity_examples=severity_examples,
        prior_str=_prior_str(severity_priors),
        severity_list=SEVERITY_LIST_STR,
    )


# ── Row 2b ─────────────────────────────────────────────────────────────────────
def build_description_prompt(clips, law12_context: str) -> str:
    return DESCRIPTION_FIRST_DESCRIPTION_TMPL.format(n_views=len(clips))


def build_severity_from_description_prompt(
    action_class: str, description: str, law12_context: str
) -> str:
    safe_action = action_class.replace("{", "{{").replace("}", "}}")
    safe_description = description.replace("{", "{{").replace("}", "}}")
    safe_law12 = law12_context.replace("{", "{{").replace("}", "}}")
    return DESCRIPTION_FIRST_SEVERITY_TMPL.format(
        action_class=safe_action,
        description=safe_description,
        law12_context=safe_law12,
        severity_list=SEVERITY_LIST_STR,
    )


# ── Row 3 ──────────────────────────────────────────────────────────────────────
def build_ragicl_prompt(n_views: int, law12_context: str, dynamic_examples: str) -> str:
    return RAGICL_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        dynamic_examples=dynamic_examples,
        action_list=ACTION_LIST_STR,
        severity_list=SEVERITY_LIST_STR,
    )


# ── Row 4 ──────────────────────────────────────────────────────────────────────
def build_cos_frame_selection_prompt(n_views: int, frames_per_view: int) -> str:
    view_labels = ["Live camera"] + [f"Replay {i}" for i in range(1, n_views)]
    view_list = "\n".join(
        f'  - "{label}": a number from 0 to {frames_per_view - 1}'
        for label in view_labels
    )
    frame_json_template = ", ".join(
        f'"{label}": <0-{frames_per_view - 1}>' for label in view_labels
    )
    return COS_FRAME_SELECT_TMPL.format(
        n_views=n_views,
        frames_per_view=frames_per_view,
        max_frame_idx=frames_per_view - 1,
        view_list=view_list,
        frame_json_template=frame_json_template,
    )


def build_cos_action_prompt(
    n_views: int, law12_context: str, mined_examples: str, selected_frame_info: str
) -> str:
    return COS_ACTION_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        mined_examples=mined_examples,
        selected_frame_info=selected_frame_info,
        action_list=ACTION_LIST_STR,
    )


def build_cos_severity_prompt(
    n_views: int,
    law12_context: str,
    predicted_action: str,
    severity_examples: str,
    severity_priors: dict,
    selected_frame_info: str,
) -> str:
    return COS_SEVERITY_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        predicted_action=predicted_action,
        severity_examples=severity_examples,
        prior_str=_prior_str(severity_priors),
        selected_frame_info=selected_frame_info,
        severity_list=SEVERITY_LIST_STR,
    )


# ── NEW Row 5: Full-frame severity ────────────────────────────────────────────
def build_full_frame_severity_prompt(
    n_views: int,
    law12_context: str,
    predicted_action: str,
    severity_examples: str,
    severity_priors: dict,
) -> str:
    """
    For severity stage: uses ALL frames (not just key frames).
    The approach trajectory and aftermath encode force information
    that a single contact frame cannot provide.
    """
    return FULL_FRAME_SEVERITY_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        predicted_action=predicted_action,
        severity_examples=severity_examples,
        prior_str=_prior_str(severity_priors),
        severity_list=SEVERITY_LIST_STR,
    )


# ── NEW Row 6: Per-action severity prior ─────────────────────────────────────
def build_per_action_prior_prompt(
    n_views: int, law12_context: str, per_action_priors: dict
) -> str:
    """
    static_few_shot + per-action severity calibration.
    Shows the model that e.g. Elbowing is almost always Red card,
    while Standing tackling is rarely Red card.
    """
    return PER_ACTION_PRIOR_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        examples=STATIC_EXAMPLES,
        per_action_prior_str=_per_action_prior_str(per_action_priors),
        action_list=ACTION_LIST_STR,
        severity_list=SEVERITY_LIST_STR,
    )


# ── NEW Row 7: Targeted retrieval ─────────────────────────────────────────────
def build_targeted_retrieval_prompt(
    n_views: int,
    law12_context: str,
    targeted_examples: str,
    predicted_action_hint: str = "this action",
) -> str:
    """
    Shows targeted examples: same action + confusable action boundary cases.
    Fixes data_driven's problem of showing too many irrelevant examples.
    """
    return TARGETED_RETRIEVAL_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        targeted_examples=targeted_examples,
        predicted_action_hint=predicted_action_hint,
        action_list=ACTION_LIST_STR,
        severity_list=SEVERITY_LIST_STR,
    )


# ── NEW Row 8: Ordinal severity ───────────────────────────────────────────────
def build_ordinal_severity_prompt(
    n_views: int,
    law12_context: str,
    predicted_action: str,
    severity_examples: str,
    severity_priors: dict,
) -> str:
    """
    Reframes severity as a step-by-step ordinal decision (0→3).
    Forces the model to reason through contact → force → recklessness → excess.
    """
    return ORDINAL_SEVERITY_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        predicted_action=predicted_action,
        severity_examples=severity_examples,
        prior_str=_prior_str(severity_priors),
        severity_list=SEVERITY_LIST_STR,
    )


# ── NEW Row 10: cos_disambig ──────────────────────────────────────────────────
def build_cos_action_disambig_prompt(
    n_views: int, law12_context: str, mined_examples: str, selected_frame_info: str
) -> str:
    """
    CoS action prompt with explicit disambiguation of Elbowing vs Holding/Pushing.
    Addresses the confusion matrix collapse where 131/132 Standing tackling
    and 46/50 Holding are predicted as Elbowing.
    """
    from .templates import ACTION_DISAMBIGUATION

    return COS_ACTION_TMPL.format(
        n_views=n_views,
        law12_context=law12_context + "\n\n" + ACTION_DISAMBIGUATION,
        mined_examples=mined_examples,
        selected_frame_info=selected_frame_info,
        action_list=ACTION_LIST_STR,
    )


def build_cos_static_severity_prompt(
    n_views: int,
    law12_context: str,
    predicted_action: str,
    severity_priors: dict,
) -> str:
    """
    CoS severity stage using static examples (not medoid retrieval).
    Uses FULL_FRAME_SEVERITY_TMPL with STATIC_EXAMPLES for severity context.
    Includes per-action hint for calibration.
    """

    def _prior_str(priors: dict) -> str:
        if not priors:
            return "No priors available."
        total = sum(priors.values())
        return "; ".join(f"{sev}: {100.0*ct/total:.1f}%" for sev, ct in priors.items())

    return FULL_FRAME_SEVERITY_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        predicted_action=predicted_action,
        severity_examples=STATIC_EXAMPLES,
        prior_str=_prior_str(severity_priors),
        severity_list=SEVERITY_LIST_STR,
    )


def build_contrastive_severity_prompt(
    n_views: int,
    law12_context: str,
    action: str,
    anchor_a_sev: str,
    anchor_b_sev: str,
) -> str:
    return CONTRASTIVE_SEVERITY_TMPL.format(
        n_views=n_views,
        law12_context=law12_context,
        action=action,
        anchor_a_sev=anchor_a_sev,
        anchor_b_sev=anchor_b_sev,
    )


def build_physics_flow_prompt(action: str, max_disp: float, severity_list: str) -> str:
    return PHYSICS_FLOW_TMPL.format(
        action=action, max_disp=max_disp, severity_list=severity_list
    )
