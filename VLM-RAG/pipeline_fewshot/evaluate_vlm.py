"""
evaluate_vlm.py
===============
Evaluates VLM-based foul classification across all prompting strategies.

Produces:
  - Per-strategy accuracy and balanced accuracy for action + severity
  - Leaderboard value (mean of balanced accuracies)
  - Per-class breakdown
  - Confusion matrices
  - results.json with all predictions

Usage:
  python evaluate_vlm.py \
    --hdf5_path /net/tscratch/people/plgaszos/SoccerNet_HDF5/Valid.hdf5 \
    --annotations /net/tscratch/people/plgaszos/SoccerNet_Data/Valid/annotations.json \
    --backend qwen \
    --strategies zero_shot rule_grounded chain_of_thought few_shot \
    --law12_pdf law12.pdf \
    --frames_per_view 4 \
    --output_dir vlm_results \
    --max_samples 100
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

import h5py
from tqdm import tqdm

from vlm_classifier import (
    VLMFoulClassifier,
    extract_keyframes,
    ACTION_CLASSES,
    SEVERITY_CLASSES,
    ACTION_TO_IDX,
    SEVERITY_TO_IDX,
)

# ---------------------------------------------------------------------------
# Label loading (mirrors your data_loader.py logic)
# ---------------------------------------------------------------------------

OFFENCE_SEVERITY_MAP = {
    ("No offence", ""): 0,
    ("No Offence", ""): 0,
    ("Offence", "1.0"): 1,
    ("Offence", "3.0"): 2,
    ("Offence", "5.0"): 3,
}

ACTION_FILTER = {"Dont know", ""}
OFFENCE_FILTER = {"Between", ""}
SEVERITY_FILTER = {"2.0", "4.0"}


def load_annotations(annotations_path: str):
    """
    Load valid annotations matching your dataset filtering logic.

    Returns dict: action_id -> {"action": int, "severity": int, "clips": [str]}
    """
    with open(annotations_path) as f:
        data = json.load(f)

    samples = {}
    for action_id, action_data in data["Actions"].items():
        action_class = action_data.get("Action class", "")
        offence_class = action_data.get("Offence", "")
        severity_class = action_data.get("Severity", "")

        # Apply same filters as your dataset
        if action_class in ACTION_FILTER:
            continue
        if (offence_class in OFFENCE_FILTER) and action_class != "Dive":
            continue
        if (
            (severity_class in SEVERITY_FILTER)
            and action_class != "Dive"
            and offence_class not in ("No offence", "No Offence")
        ):
            continue

        # Fix borderline cases
        if offence_class in OFFENCE_FILTER:
            offence_class = "Offence"
        if severity_class in SEVERITY_FILTER:
            severity_class = "1.0"

        # Map to severity index
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
            "action": action_idx,
            "severity": severity_idx,
            "clips": clips,
        }

    return samples


# ---------------------------------------------------------------------------
# Metrics (matching your SoccerNet evaluator)
# ---------------------------------------------------------------------------


def compute_metrics(y_true_action, y_pred_action, y_true_severity, y_pred_severity):
    # Filter invalid predictions (-1 = parse failed)
    valid = [
        i
        for i in range(len(y_pred_action))
        if y_pred_action[i] != -1 and y_pred_severity[i] != -1
    ]
    n_valid = len(valid)
    n_total = len(y_pred_action)

    if n_valid == 0:
        return {"error": "No valid predictions"}

    ya_true = [y_true_action[i] for i in valid]
    ya_pred = [y_pred_action[i] for i in valid]
    ys_true = [y_true_severity[i] for i in valid]
    ys_pred = [y_pred_severity[i] for i in valid]

    acc_action = accuracy_score(ya_true, ya_pred) * 100
    bacc_action = balanced_accuracy_score(ya_true, ya_pred) * 100
    acc_sev = accuracy_score(ys_true, ys_pred) * 100
    bacc_sev = balanced_accuracy_score(ys_true, ys_pred) * 100
    lb_value = (bacc_action + bacc_sev) / 2

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "parse_rate": n_valid / n_total * 100,
        "accuracy_action": acc_action,
        "balanced_acc_action": bacc_action,
        "accuracy_severity": acc_sev,
        "balanced_acc_severity": bacc_sev,
        "leaderboard_value": lb_value,
        "confusion_action": confusion_matrix(
            ya_true, ya_pred, labels=list(range(len(ACTION_CLASSES)))
        ).tolist(),
        "confusion_severity": confusion_matrix(
            ys_true, ys_pred, labels=list(range(len(SEVERITY_CLASSES)))
        ).tolist(),
    }


# ---------------------------------------------------------------------------
# Single-strategy evaluation
# ---------------------------------------------------------------------------


def evaluate_strategy(
    classifier: VLMFoulClassifier,
    hdf5_path: str,
    samples: dict,
    max_samples: int = None,
    frames_per_view: int = 4,
) -> dict:
    """Run evaluation for a single strategy."""

    y_true_action, y_pred_action = [], []
    y_true_severity, y_pred_severity = [], []
    predictions = {}

    sample_ids = list(samples.keys())
    if max_samples:
        sample_ids = sample_ids[:max_samples]

    with h5py.File(hdf5_path, "r", swmr=True) as hdf5:
        for action_id in tqdm(sample_ids, desc=f"  [{classifier.strategy}]"):
            sample = samples[action_id]
            action_key = f"action_{action_id}"

            # Load frames for each available clip
            frames_per_view_list = []
            for clip_key in sample["clips"][:4]:  # max 4 views
                clip_name = clip_key.replace(".mp4", "")
                frames = extract_keyframes(
                    hdf5,
                    action_key,
                    clip_name,
                    n_frames=frames_per_view,
                )
                if frames:
                    frames_per_view_list.append(frames)

            if not frames_per_view_list:
                # No frames loaded — mark as parse failure
                y_true_action.append(sample["action"])
                y_pred_action.append(-1)
                y_true_severity.append(sample["severity"])
                y_pred_severity.append(-1)
                continue

            # Classify
            try:
                action_idx, severity_idx, raw = classifier.classify_action(
                    frames_per_view=frames_per_view_list,
                    action_hint=ACTION_CLASSES[sample["action"]],
                )
            except Exception as e:
                print(f"  Error on {action_id}: {e}")
                action_idx, severity_idx, raw = -1, -1, str(e)

            y_true_action.append(sample["action"])
            y_pred_action.append(action_idx)
            y_true_severity.append(sample["severity"])
            y_pred_severity.append(severity_idx)

            predictions[action_id] = {
                "true_action": sample["action"],
                "pred_action": action_idx,
                "true_severity": sample["severity"],
                "pred_severity": severity_idx,
                "raw_response": raw,
            }

    metrics = compute_metrics(
        y_true_action,
        y_pred_action,
        y_true_severity,
        y_pred_severity,
    )
    metrics["predictions"] = predictions
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading annotations...")
    samples = load_annotations(args.annotations)
    print(f"Found {len(samples)} valid samples.")

    strategies = args.strategies
    all_results = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        classifier = VLMFoulClassifier(
            backend=args.backend,
            strategy=strategy,
            law12_pdf=args.law12_pdf,
            frames_per_view=args.frames_per_view,
        )

        metrics = evaluate_strategy(
            classifier=classifier,
            hdf5_path=args.hdf5_path,
            samples=samples,
            max_samples=args.max_samples,
            frames_per_view=args.frames_per_view,
        )
        all_results[strategy] = metrics

        # Save per-strategy results
        strat_path = output_dir / f"{strategy}_results.json"
        with open(strat_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Print summary
        print(f"\nResults for {strategy}:")
        print(f"  Parse rate:          {metrics.get('parse_rate', 0):.1f}%")
        print(f"  Action accuracy:     {metrics.get('accuracy_action', 0):.2f}%")
        print(f"  Action balanced:     {metrics.get('balanced_acc_action', 0):.2f}%")
        print(f"  Severity accuracy:   {metrics.get('accuracy_severity', 0):.2f}%")
        print(f"  Severity balanced:   {metrics.get('balanced_acc_severity', 0):.2f}%")
        print(f"  Leaderboard value:   {metrics.get('leaderboard_value', 0):.4f}")

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'Strategy':<20} {'Parse%':>7} {'Act BA':>8} {'Sev BA':>8} {'LB Val':>8}")
    print("-" * 55)
    for s, m in all_results.items():
        print(
            f"{s:<20} {m.get('parse_rate',0):>7.1f} "
            f"{m.get('balanced_acc_action',0):>8.2f} "
            f"{m.get('balanced_acc_severity',0):>8.2f} "
            f"{m.get('leaderboard_value',0):>8.4f}"
        )

    # Save full results
    summary_path = output_dir / "all_results.json"
    # Remove predictions from summary to keep it readable
    summary = {
        s: {k: v for k, v in m.items() if k != "predictions"}
        for s, m in all_results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument(
        "--backend", default="qwen", choices=["qwen", "internvl", "gpt4o"]
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero_shot", "rule_grounded", "chain_of_thought", "few_shot"],
    )
    parser.add_argument("--law12_pdf", default=None)
    parser.add_argument("--frames_per_view", type=int, default=4)
    parser.add_argument("--output_dir", default="vlm_results")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit samples for quick testing (None = all)",
    )
    args = parser.parse_args()
    main(args)
