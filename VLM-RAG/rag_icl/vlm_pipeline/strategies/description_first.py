"""
strategies/description_first.py
===============================
Three-stage strategy that first extracts a factual description of the contact,
then classifies the action, then infers severity from the description only.
"""

from typing import List, Optional, Tuple

from PIL import Image

from .base import parse_action_only, parse_response, parse_severity_only
from ..utils.constants import ACTION_CLASSES
from ..utils.frames import (
    parse_key_frames,
    select_key_frames,
    format_selected_frame_info,
)
from ..prompts import (
    build_description_prompt,
    build_severity_from_description_prompt,
    build_static_prompt,
    build_cos_frame_selection_prompt,
    build_cos_action_prompt,
)
from ..retrieval.medoid_cache import build_examples_text


def _select_description_frames(
    frames_per_view: List[List[Image.Image]],
) -> List[List[Image.Image]]:
    selected_views: List[List[Image.Image]] = []
    for frames in frames_per_view:
        if not frames:
            continue
        candidate_indices = [0, len(frames) // 2, len(frames) - 1]
        seen = []
        for idx in candidate_indices:
            idx = max(0, min(idx, len(frames) - 1))
            if idx not in seen:
                seen.append(idx)
        selected_views.append([frames[idx] for idx in seen])
    return selected_views


class DescriptionFirstStrategy:
    def classify(
        self,
        backend,
        frames_per_view: List[List[Image.Image]],
        law12_ctx: str,
        medoid_cache: Optional[dict] = None,
        frames_per_view_count: int = 4,
    ) -> Tuple[int, int, str]:
        description_frames = _select_description_frames(frames_per_view)
        description_prompt = build_description_prompt(description_frames, law12_ctx)
        raw_description = backend.classify(description_frames, description_prompt)
        description = raw_description.strip()

        n_views = len(frames_per_view)
        if medoid_cache:
            sel_prompt = build_cos_frame_selection_prompt(
                n_views, frames_per_view_count
            )
            raw0 = backend.classify(frames_per_view, sel_prompt)
            key_indices = parse_key_frames(raw0, n_views, frames_per_view_count)
            key_frames = select_key_frames(
                frames_per_view, key_indices, context_window=1
            )
            sel_info = format_selected_frame_info(key_indices, n_views)

            all_text, all_imgs = build_examples_text(medoid_cache, n_per_class=1)
            act_prompt = build_cos_action_prompt(
                n_views=n_views,
                law12_context=law12_ctx,
                mined_examples=all_text,
                selected_frame_info=sel_info,
            )
            raw1 = backend.classify(key_frames, act_prompt, extra_images=all_imgs)
            act_idx = parse_action_only(raw1)
            act_raw = raw1
        else:
            act_prompt = build_static_prompt(n_views, law12_ctx)
            raw1 = backend.classify(frames_per_view, act_prompt)
            act_idx, _ = parse_response(raw1)
            act_raw = raw1

        act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"
        sev_prompt = build_severity_from_description_prompt(
            action_class=act_str,
            description=description,
            law12_context=law12_ctx,
        )
        raw2 = backend.classify([], sev_prompt)
        sev_idx = parse_severity_only(raw2)

        raw = (
            f"STAGE1 (description): {raw_description}\n"
            f"STAGE2 (action): {act_raw}\n"
            f"STAGE3 (severity from description): {raw2}"
        )
        return act_idx, sev_idx, raw
