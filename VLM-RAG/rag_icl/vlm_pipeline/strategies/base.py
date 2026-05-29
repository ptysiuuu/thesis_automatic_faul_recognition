"""
strategies/base.py
==================
Shared parsing utilities used by all strategies.
"""

import re
import json
from typing import Tuple, List
from ..utils.constants import ACTION_CLASSES, SEVERITY_CLASSES


def _extract_json_candidates(text: str) -> List[str]:
    candidates = []
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        i = start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
            i += 1
    # Prefer larger JSON blocks first (outer objects)
    candidates = sorted(set(candidates), key=len, reverse=True)
    return candidates


def _sanitize_json_candidate(candidate: str) -> str:
    """Fix unescaped newlines and tabs inside JSON string values. Mistral sometimes outputs literal newlines in strings."""
    res = []
    in_string = False
    escape = False
    for char in candidate:
        if char == '"' and not escape:
            in_string = not in_string

        if in_string:
            if char == "\n":
                res.append("\\n")
                escape = False
                continue
            elif char == "\t":
                res.append("\\t")
                escape = False
                continue
            elif char == "\r":
                res.append("\\r")
                escape = False
                continue

        if char == "\\":
            escape = not escape
        else:
            escape = False

        res.append(char)
    return "".join(res)


def parse_response(text: str) -> Tuple[int, int]:
    """Parse VLM JSON response into (action_idx, severity_idx). Returns (-1,-1) on failure."""
    text = re.sub(r"```json\s*|\s*```", "", text).strip()
    candidates = _extract_json_candidates(text)
    if not candidates:
        return -1, -1

    best_action = -1
    best_severity = -1

    for candidate in candidates:
        try:
            can_sanitized = _sanitize_json_candidate(candidate)
            data = json.loads(can_sanitized)
        except json.JSONDecodeError:
            try:
                data = json.loads(can_sanitized.replace("'", '"'))
            except Exception:
                continue

        if not isinstance(data, dict):
            continue

        a_str = data.get("action", "")
        s_str = data.get("severity", "")

        if isinstance(a_str, str):
            best_action = next(
                (
                    i
                    for i, a in enumerate(ACTION_CLASSES)
                    if a.lower() in a_str.lower() or a_str.lower() in a.lower()
                ),
                best_action,
            )

        if "severity_level" in data:
            level = data.get("severity_level", -1)
            try:
                level = int(level)
                if 0 <= level <= 3:
                    best_severity = level
            except (TypeError, ValueError):
                pass

        if isinstance(s_str, str):
            best_severity = next(
                (
                    i
                    for i, s in enumerate(SEVERITY_CLASSES)
                    if s.lower() in s_str.lower() or s_str.lower() in s.lower()
                ),
                best_severity,
            )

        if best_action != -1 and best_severity != -1:
            return best_action, best_severity

    if best_action == -1:
        m = re.search(
            r'["\']action["\']\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE
        )
        if m:
            a_str = m.group(1)
            best_action = next(
                (
                    i
                    for i, a in enumerate(ACTION_CLASSES)
                    if a.lower() in a_str.lower() or a_str.lower() in a.lower()
                ),
                best_action,
            )

    if best_severity == -1:
        m = re.search(r'["\']severity_level["\']\s*:\s*(\d+)', text, re.IGNORECASE)
        if m:
            level = int(m.group(1))
            if 0 <= level <= 3:
                best_severity = level
        else:
            m = re.search(
                r'["\']severity["\']\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE
            )
            if m:
                s_str = m.group(1)
                best_severity = next(
                    (
                        i
                        for i, s in enumerate(SEVERITY_CLASSES)
                        if s.lower() in s_str.lower() or s_str.lower() in s.lower()
                    ),
                    best_severity,
                )

    return best_action, best_severity


def parse_action_only(text: str) -> int:
    text = re.sub(r"```json\s*|\s*```", "", text).strip()
    candidates = _extract_json_candidates(text)
    if not candidates:
        return -1
    for candidate in candidates:
        try:
            data = json.loads(_sanitize_json_candidate(candidate))
        except Exception:
            try:
                data = json.loads(_sanitize_json_candidate(candidate).replace("'", '"'))
            except Exception:
                continue
        if not isinstance(data, dict) or "action" not in data:
            continue
        a_str = data.get("action", "")
        if isinstance(a_str, str):
            idx = next(
                (
                    i
                    for i, a in enumerate(ACTION_CLASSES)
                    if a.lower() in a_str.lower() or a_str.lower() in a.lower()
                ),
                -1,
            )
            if idx != -1:
                return idx

    # Regex fallback
    m = re.search(r'["\']action["\']\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    if m:
        a_str = m.group(1)
        idx = next(
            (
                i
                for i, a in enumerate(ACTION_CLASSES)
                if a.lower() in a_str.lower() or a_str.lower() in a.lower()
            ),
            -1,
        )
        if idx != -1:
            return idx

    return -1


def parse_severity_only(text: str) -> int:
    text = re.sub(r"```json\s*|\s*```", "", text).strip()
    candidates = _extract_json_candidates(text)
    if not candidates:
        return -1
    for candidate in candidates:
        try:
            data = json.loads(_sanitize_json_candidate(candidate))
        except Exception:
            try:
                data = json.loads(_sanitize_json_candidate(candidate).replace("'", '"'))
            except Exception:
                continue

        if not isinstance(data, dict):
            continue

        # Handle ordinal severity format: {"severity_level": 2, "severity": "Yellow card"}
        if "severity_level" in data:
            level = data.get("severity_level", -1)
            try:
                level = int(level)
                if 0 <= level <= 3:
                    return level
            except (TypeError, ValueError):
                pass

        s_str = data.get("severity", "")
        if isinstance(s_str, str):
            idx = next(
                (
                    i
                    for i, s in enumerate(SEVERITY_CLASSES)
                    if s.lower() in s_str.lower() or s_str.lower() in s.lower()
                ),
                -1,
            )
            if idx != -1:
                return idx

    m = re.search(r'["\']severity_level["\']\s*:\s*(\d+)', text, re.IGNORECASE)
    if m:
        level = int(m.group(1))
        if 0 <= level <= 3:
            return level

    m = re.search(r'["\']severity["\']\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    if m:
        s_str = m.group(1)
        idx = next(
            (
                i
                for i, s in enumerate(SEVERITY_CLASSES)
                if s.lower() in s_str.lower() or s_str.lower() in s.lower()
            ),
            -1,
        )
        if idx != -1:
            return idx

    return -1
