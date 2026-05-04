"""
strategies/base.py
==================
Shared parsing utilities used by all strategies.
"""

import re
import json
from typing import Tuple
from ..utils.constants import ACTION_CLASSES, SEVERITY_CLASSES


def parse_response(text: str) -> Tuple[int, int]:
    """Parse VLM JSON response into (action_idx, severity_idx). Returns (-1,-1) on failure."""
    text = re.sub(r"```json\s*|\s*```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return -1, -1
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        try:
            data = json.loads(match.group().replace("'", '"'))
        except Exception:
            return -1, -1

    a_str = data.get("action", "")
    s_str = data.get("severity", "")

    action_idx = next(
        (i for i, a in enumerate(ACTION_CLASSES)
         if a.lower() in a_str.lower() or a_str.lower() in a.lower()), -1)
    severity_idx = next(
        (i for i, s in enumerate(SEVERITY_CLASSES)
         if s.lower() in s_str.lower() or s_str.lower() in s.lower()), -1)
    return action_idx, severity_idx


def parse_action_only(text: str) -> int:
    text = re.sub(r"```json\s*|\s*```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return -1
    try:
        data = json.loads(match.group())
    except Exception:
        return -1
    a_str = data.get("action", "")
    return next(
        (i for i, a in enumerate(ACTION_CLASSES)
         if a.lower() in a_str.lower() or a_str.lower() in a.lower()), -1)


def parse_severity_only(text: str) -> int:
    text = re.sub(r"```json\s*|\s*```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return -1
    try:
        data = json.loads(match.group())
    except Exception:
        return -1

    # Handle ordinal severity format: {"severity_level": 2, "severity": "Yellow card"}
    if "severity_level" in (data if isinstance(data, dict) else {}):
        level = data.get("severity_level", -1)
        try:
            level = int(level)
            if 0 <= level <= 3:
                return level
        except (TypeError, ValueError):
            pass

    s_str = data.get("severity", "")
    return next(
        (i for i, s in enumerate(SEVERITY_CLASSES)
         if s.lower() in s_str.lower() or s_str.lower() in s.lower()), -1)
