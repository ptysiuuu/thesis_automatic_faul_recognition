"""
evaluate_ragicl.py
==================
Evaluates all four strategies on the validation or test set and
produces the comparison table that is the core thesis result.

Strategies:
  zero_shot       — baseline, no context
  rule_grounded   — FIFA Law 12 RAG only
  static_few_shot — fixed handcrafted examples + rules
  rag_icl         — dynamic MViT-retrieved examples + rules (NOVEL)

Usage:
  python evaluate_ragicl.py \
    --hdf5_path        /net/tscratch/people/plgaszos/SoccerNet_HDF5/Valid.hdf5 \
    --annotations      /net/tscratch/people/plgaszos/SoccerNet_Data/Valid/annotations.json \
    --law12_pdf        /net/tscratch/people/plgaszos/SoccerNet_Data/law12.pdf \
    --faiss_index_path /net/tscratch/people/plgaszos/vlm_rag_icl/train_mvit_features.index \
    --faiss_meta_path  /net/tscratch/people/plgaszos/vlm_rag_icl/train_mvit_metadata.json \
    --strategies       zero_shot rule_grounded static_few_shot rag_icl \
    --frames_per_view  4 \
    --output_dir       ragicl_results \
    --max_samples      50
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import h5py
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

from vlm_classifier_ragicl import (
    VLMFoulClassifier, extract_keyframes, parse_response,
    ACTION_CLASSES, SEVERITY_CLASSES, ACTION_TO_IDX, SEVERITY_TO_IDX,
)


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

OFFENCE_SEVERITY_MAP = {
    ("No offence", ""): 0, ("No Offence", ""): 0,
    ("Offence", "1.0"): 1, ("Offence", "3.0"): 2, ("Offence", "5.0"): 3,
}


def load_annotations(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)

    samples = {}
    for action_id, action_data in data["Actions"].items():
        action_class  = action_data.get("Action class", "")
        offence_class = action_data.get("Offence", "")
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true_a, y_pred_a, y_true_s, y_pred_s) -> dict:
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
        "n_total":              n_total,
        "n_valid":              n_valid,
        "parse_rate":           n_valid / n_total * 100,
        "accuracy_action":      accuracy_score(ya_true, ya_pred) * 100,
        "balanced_acc_action":  balanced_accuracy_score(ya_true, ya_pred) * 100,
        "accuracy_severity":    accuracy_score(ys_true, ys_pred) * 100,
        "balanced_acc_severity":balanced_accuracy_score(ys_true, ys_pred) * 100,
        "leaderboard_value":    (balanced_accuracy_score(ya_true, ya_pred) +
                                 balanced_accuracy_score(ys_true, ys_pred)) / 2 * 100,
        "confusion_action":     confusion_matrix(
            ya_true, ya_pred, labels=list(range(len(ACTION_CLASSES)))).tolist(),
        "confusion_severity":   confusion_matrix(
            ys_true, ys_pred, labels=list(range(len(SEVERITY_CLASSES)))).tolist(),
    }


# ---------------------------------------------------------------------------
# Single-strategy evaluation loop
# ---------------------------------------------------------------------------

def evaluate_strategy(classifier: VLMFoulClassifier,
                       hdf5_path: str, samples: dict,
                       max_samples: int = None,
                       frames_per_view: int = 4) -> dict:
    y_true_a, y_pred_a = [], []
    y_true_s, y_pred_s = [], []
    predictions = {}

    sample_ids = list(samples.keys())
    if max_samples:
        sample_ids = sample_ids[:max_samples]

    with h5py.File(hdf5_path, "r", swmr=True) as hdf5:
        for action_id in tqdm(sample_ids, desc=f"  [{classifier.strategy}]"):
            sample     = samples[action_id]
            action_key = f"action_{action_id}"

            frames_per_view_list = []
            for clip_key in sample["clips"][:4]:
                clip_name = clip_key.replace(".mp4", "")
                frames = extract_keyframes(hdf5, action_key, clip_name,
                                           n_frames=frames_per_view)
                if frames:
                    frames_per_view_list.append(frames)

            if not frames_per_view_list:
                y_true_a.append(sample["action"]); y_pred_a.append(-1)
                y_true_s.append(sample["severity"]); y_pred_s.append(-1)
                continue

            try:
                act_idx, sev_idx, raw = classifier.classify_action(
                    frames_per_view=frames_per_view_list,
                    action_hint=ACTION_CLASSES[sample["action"]],
                )
            except Exception as e:
                print(f"  Error on {action_id}: {e}")
                act_idx, sev_idx, raw = -1, -1, str(e)

            y_true_a.append(sample["action"]); y_pred_a.append(act_idx)
            y_true_s.append(sample["severity"]); y_pred_s.append(sev_idx)
            predictions[action_id] = {
                "true_action":   sample["action"],   "pred_action":   act_idx,
                "true_severity": sample["severity"], "pred_severity": sev_idx,
                "raw_response":  raw,
            }

    metrics = compute_metrics(y_true_a, y_pred_a, y_true_s, y_pred_s)
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

    all_results = {}

    for strategy in args.strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        classifier = VLMFoulClassifier(
            backend          = args.backend,
            strategy         = strategy,
            law12_pdf        = args.law12_pdf,
            frames_per_view  = args.frames_per_view,
            faiss_index_path = args.faiss_index_path,
            faiss_meta_path  = args.faiss_meta_path,
            retrieval_k      = args.retrieval_k,
        )

        metrics = evaluate_strategy(
            classifier=classifier,
            hdf5_path=args.hdf5_path,
            samples=samples,
            max_samples=args.max_samples,
            frames_per_view=args.frames_per_view,
        )
        all_results[strategy] = metrics

        # Save per-strategy JSON
        with open(output_dir / f"{strategy}_results.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n  Parse rate:       {metrics.get('parse_rate', 0):.1f}%")
        print(f"  Action BA:        {metrics.get('balanced_acc_action', 0):.2f}%")
        print(f"  Severity BA:      {metrics.get('balanced_acc_severity', 0):.2f}%")
        print(f"  Leaderboard:      {metrics.get('leaderboard_value', 0):.4f}")

    # ── Comparison table ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("COMPARISON TABLE")
    print(f"{'Strategy':<20} {'Parse%':>7} {'Act BA':>8} {'Sev BA':>8} {'LB Val':>8}")
    print("-" * 55)
    for s, m in all_results.items():
        print(f"{s:<20} {m.get('parse_rate',0):>7.1f} "
              f"{m.get('balanced_acc_action',0):>8.2f} "
              f"{m.get('balanced_acc_severity',0):>8.2f} "
              f"{m.get('leaderboard_value',0):>8.4f}")

    # Save summary (without raw predictions)
    summary = {s: {k: v for k, v in m.items() if k not in
                   ("predictions", "confusion_action", "confusion_severity")}
               for s, m in all_results.items()}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path",        required=True)
    parser.add_argument("--annotations",      required=True)
    parser.add_argument("--backend",          default="qwen")
    parser.add_argument("--strategies",       nargs="+",
                        default=["zero_shot", "rule_grounded",
                                 "static_few_shot", "rag_icl"])
    parser.add_argument("--law12_pdf",        default=None)
    parser.add_argument("--faiss_index_path", default=None)
    parser.add_argument("--faiss_meta_path",  default=None)
    parser.add_argument("--retrieval_k",      type=int, default=3)
    parser.add_argument("--frames_per_view",  type=int, default=4)
    parser.add_argument("--output_dir",       default="ragicl_results")
    parser.add_argument("--max_samples",      type=int, default=None)
    args = parser.parse_args()
    main(args)
