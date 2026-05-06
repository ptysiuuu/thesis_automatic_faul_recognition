"""
conditional_ensemble.py
=======================
Conditional calibration for severity using VLM action + MVNet severity.

Example:
  python conditional_ensemble.py \
    --vlm_results ablation_results/cos_two_stage/results.json \
    --mvnet_preds VARS_model/VARS_step10/predicitions_valid_epoch_28.json \
    --annotations SoccerNet_Data/Valid/annotations.json \
    --output ensemble_conditional_results.json \
    --export_sn ensemble_conditional_predictions.json
"""

import argparse
import json
from collections import Counter

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from exc


ACTION_CLASSES = [
    "Tackling",
    "Standing tackling",
    "High leg",
    "Holding",
    "Pushing",
    "Elbowing",
    "Challenge",
    "Dive",
]

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


def load_annotations(annotations_path):
    with open(annotations_path) as f:
        data = json.load(f)

    samples = {}
    for action_id, action_data in data["Actions"].items():
        action_class = action_data.get("Action class", "")
        offence_class = action_data.get("Offence", "")
        severity_class = action_data.get("Severity", "")

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

        if offence_class in OFFENCE_FILTER:
            offence_class = "Offence"
        if severity_class in SEVERITY_FILTER:
            severity_class = "1.0"

        key = (offence_class, severity_class)
        if key in OFFENCE_SEVERITY_MAP:
            severity_idx = OFFENCE_SEVERITY_MAP[key]
        elif offence_class in ("No offence", "No Offence"):
            severity_idx = 0
        else:
            continue

        if action_class not in ACTION_CLASSES:
            continue
        action_idx = ACTION_CLASSES.index(action_class)

        samples[action_id] = {
            "action": action_idx,
            "severity": severity_idx,
        }

    return samples


def load_vlm_predictions(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("predictions", data)


def load_mvnet_predictions(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("Actions", data)


def mvnet_pred_severity(entry):
    offence = entry.get("Offence", "")
    severity = entry.get("Severity", "")
    if (offence, severity) in OFFENCE_SEVERITY_MAP:
        return OFFENCE_SEVERITY_MAP[(offence, severity)]
    if offence in ("No offence", "No Offence"):
        return 0
    return None


def one_hot(idx, size):
    vec = [0.0] * size
    vec[idx] = 1.0
    return vec


def get_act_features(vlm_entry, use_logits):
    if use_logits:
        act_logits = vlm_entry.get("act_logits")
        if isinstance(act_logits, list) and len(act_logits) == len(ACTION_CLASSES):
            return [float(x) for x in act_logits]
    act_idx = vlm_entry.get("pred_action")
    if act_idx is None:
        return None
    if act_idx < 0 or act_idx >= len(ACTION_CLASSES):
        return None
    return one_hot(act_idx, len(ACTION_CLASSES))


def get_sev_features(mvnet_entry, pred_sev, use_logits):
    if use_logits:
        sev_logits = mvnet_entry.get("sev_logits")
        if isinstance(sev_logits, list) and len(sev_logits) in (3, 4):
            return [float(x) for x in sev_logits]
    return one_hot(pred_sev, 4)


def compute_scores(y_true_action, y_pred_action, y_true_sev, y_pred_sev):
    act_ba = balanced_accuracy_score(y_true_action, y_pred_action) * 100
    sev_ba = balanced_accuracy_score(y_true_sev, y_pred_sev) * 100
    lb = (act_ba + sev_ba) / 2
    return act_ba, sev_ba, lb


def train_calibrator(X, y, c_value, max_iter, seed, folds):
    class_counts = Counter(y)
    min_count = min(class_counts.values()) if class_counts else 0
    n_splits = min(folds, min_count)

    def make_model():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=max_iter, multi_class="auto"),
        )

    if n_splits < 2:
        model = make_model()
        model.fit(X, y)
        return model, model.predict(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = np.full_like(y, fill_value=-1)
    for train_idx, test_idx in skf.split(X, y):
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])

    final_model = make_model()
    final_model.fit(X, y)
    return final_model, y_pred


def export_soccer_net(predictions, output_path, set_name):
    data = {"Set": set_name, "Actions": {}}
    for action_id, pred in predictions.items():
        sev = pred["pred_severity"]
        if sev == 0:
            offence = "No offence"
            severity = ""
        elif sev == 1:
            offence = "Offence"
            severity = "1.0"
        elif sev == 2:
            offence = "Offence"
            severity = "3.0"
        else:
            offence = "Offence"
            severity = "5.0"

        action_idx = pred["pred_action"]
        data["Actions"][action_id] = {
            "Action class": ACTION_CLASSES[action_idx],
            "Offence": offence,
            "Severity": severity,
        }

    with open(output_path, "w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_results", required=True)
    parser.add_argument("--mvnet_preds", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", default="ensemble_conditional_results.json")
    parser.add_argument("--export_sn", default=None)
    parser.add_argument("--set_name", default="valid")
    parser.add_argument("--c_value", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_logits", action="store_true")
    args = parser.parse_args()

    print(f"Loading MVNet predictions from: {args.mvnet_preds}")
    mvnet_preds = load_mvnet_predictions(args.mvnet_preds)
    print(f"Loading VLM predictions from: {args.vlm_results}")
    vlm_preds = load_vlm_predictions(args.vlm_results)
    annotations = load_annotations(args.annotations)

    common_ids = sorted(
        set(vlm_preds.keys()) & set(mvnet_preds.keys()) & set(annotations.keys())
    )

    X, y = [], []
    y_true_action, y_pred_action = [], []
    y_true_sev, y_pred_sev_mvnet = [], []
    kept_ids = []

    for action_id in common_ids:
        vlm_entry = vlm_preds[action_id]
        mvnet_entry = mvnet_preds[action_id]
        ann = annotations[action_id]

        pred_sev_mvnet = mvnet_pred_severity(mvnet_entry)
        if pred_sev_mvnet is None:
            continue

        act_feats = get_act_features(vlm_entry, args.use_logits)
        if act_feats is None:
            continue

        sev_feats = get_sev_features(mvnet_entry, pred_sev_mvnet, args.use_logits)

        X.append(act_feats + sev_feats)
        y.append(ann["severity"])
        kept_ids.append(action_id)

        y_true_action.append(ann["action"])
        y_pred_action.append(vlm_entry["pred_action"])
        y_true_sev.append(ann["severity"])
        y_pred_sev_mvnet.append(pred_sev_mvnet)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print(f"Common samples: {len(kept_ids)}")
    if len(kept_ids) == 0:
        raise SystemExit("No valid samples after filtering.")

    act_ba, sev_ba, lb = compute_scores(
        y_true_action, y_pred_action, y_true_sev, y_pred_sev_mvnet
    )
    print("Baseline (VLM action + MVNet severity)")
    print(
        f"  Act BA: {act_ba:6.2f}  Sev BA: {sev_ba:6.2f}  LB: {lb:6.2f}  N={len(kept_ids)}"
    )

    model, y_pred_sev_cv = train_calibrator(
        X, y, args.c_value, args.max_iter, args.seed, args.cv_folds
    )
    act_ba_c, sev_ba_c, lb_c = compute_scores(
        y_true_action, y_pred_action, y_true_sev, y_pred_sev_cv
    )
    print("Conditional ensemble (CV severity)")
    print(
        f"  Act BA: {act_ba_c:6.2f}  Sev BA: {sev_ba_c:6.2f}  LB: {lb_c:6.2f}  N={len(kept_ids)}"
    )

    y_pred_sev_full = model.predict(X)
    predictions = {}
    for i, action_id in enumerate(kept_ids):
        predictions[action_id] = {
            "true_action": int(y_true_action[i]),
            "pred_action": int(y_pred_action[i]),
            "true_severity": int(y_true_sev[i]),
            "pred_severity": int(y_pred_sev_full[i]),
        }

    results = {
        "n_samples": int(len(kept_ids)),
        "act_ba": act_ba_c,
        "sev_ba": sev_ba_c,
        "leaderboard_value": lb_c,
        "predictions": predictions,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved conditional ensemble results to: {args.output}")

    if args.export_sn:
        export_soccer_net(predictions, args.export_sn, args.set_name)
        print(f"Saved SoccerNet predictions to: {args.export_sn}")


if __name__ == "__main__":
    main()
