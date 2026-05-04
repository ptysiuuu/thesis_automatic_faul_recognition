"""
utils/annotations.py
====================
Annotation loading (mirrors SoccerNet-MVFoul dataset filtering logic)
and metric computation matching the official leaderboard evaluator.
"""

import json
import numpy as np
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

from .constants import (
    ACTION_CLASSES, SEVERITY_CLASSES, ACTION_TO_IDX,
    OFFENCE_SEVERITY_MAP, SEVERITY_PRIOR_DEFAULT,
)


def load_annotations(path: str) -> dict:
    """
    Load valid annotations matching SoccerNet-MVFoul dataset filtering.
    Returns dict: action_id -> {"action": int, "severity": int, "clips": [str]}
    """
    with open(path) as f:
        data = json.load(f)

    samples = {}
    for action_id, action_data in data["Actions"].items():
        action_class   = action_data.get("Action class", "")
        offence_class  = action_data.get("Offence", "")
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

        clips = [c["Url"].split("/")[-1] for c in action_data.get("Clips", [])]
        samples[action_id] = {
            "action":   action_idx,
            "severity": severity_idx,
            "clips":    clips,
        }
    return samples


def compute_metrics(y_true_a, y_pred_a, y_true_s, y_pred_s) -> dict:
    """
    Compute metrics matching SoccerNet-MVFoul leaderboard evaluator.
    LB = (balanced_acc_action + balanced_acc_severity) / 2
    """
    valid = [i for i in range(len(y_pred_a))
             if y_pred_a[i] != -1 and y_pred_s[i] != -1]
    n_valid = len(valid)
    n_total = len(y_pred_a)

    if n_valid == 0:
        return {"error": "No valid predictions"}

    ya_true = [y_true_a[i] for i in valid]
    ya_pred = [y_pred_a[i] for i in valid]
    ys_true = [y_true_s[i] for i in valid]
    ys_pred = [y_pred_s[i] for i in valid]

    return {
        "n_total":               n_total,
        "n_valid":               n_valid,
        "parse_rate":            n_valid / n_total * 100,
        "accuracy_action":       accuracy_score(ya_true, ya_pred) * 100,
        "balanced_acc_action":   balanced_accuracy_score(ya_true, ya_pred) * 100,
        "accuracy_severity":     accuracy_score(ys_true, ys_pred) * 100,
        "balanced_acc_severity": balanced_accuracy_score(ys_true, ys_pred) * 100,
        "leaderboard_value":     (balanced_accuracy_score(ya_true, ya_pred) +
                                  balanced_accuracy_score(ys_true, ys_pred)) / 2 * 100,
        "confusion_action":      confusion_matrix(
            ya_true, ya_pred, labels=list(range(len(ACTION_CLASSES)))).tolist(),
        "confusion_severity":    confusion_matrix(
            ys_true, ys_pred, labels=list(range(len(SEVERITY_CLASSES)))).tolist(),
    }


def compute_severity_priors(train_annotations_path: str) -> dict:
    """Compute actual severity distribution from training set."""
    samples = load_annotations(train_annotations_path)
    counts = Counter(SEVERITY_CLASSES[s["severity"]] for s in samples.values())
    total = sum(counts.values())
    priors = {k: round(counts.get(k, 0) / total * 100) for k in SEVERITY_CLASSES}
    print(f"[Priors] Severity distribution: {priors}")
    return priors


def compute_per_action_severity_priors(train_annotations_path: str) -> dict:
    """Compute per-action severity distribution from training set."""
    from collections import defaultdict
    samples = load_annotations(train_annotations_path)

    per_action = defaultdict(Counter)
    for s in samples.values():
        action   = ACTION_CLASSES[s["action"]]
        severity = SEVERITY_CLASSES[s["severity"]]
        per_action[action][severity] += 1

    priors = {}
    for action, counts in per_action.items():
        total = sum(counts.values())
        priors[action] = {k: round(counts.get(k, 0) / total * 100)
                          for k in SEVERITY_CLASSES}
    print(f"[Priors] Per-action priors computed for {len(priors)} actions")
    return priors
