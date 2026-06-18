"""
vlm_multiturn_var.py
====================
Multi-Turn VAR: structured cross-view reasoning without any labelled training data.

Thesis contribution experiment. Each test clip is processed as a structured
conversation — one camera view per turn — so the model accumulates cross-view
evidence before committing to a ruling:

  Turn 1 (live camera) : observe approach, spatial context, body positioning
  Turn 2 (replay 1)    : contact type, force, body part struck, danger
  Turn 3 (replay 2+)   : confirm/refine evidence from perpendicular angle
  Final turn           : classify action + severity from accumulated evidence

The key distinction from zero-shot baselines: genuine multi-turn conversation
history (assistant responses fed back) rather than a single large prompt.

Evaluates BA_act and BA_sev on the Test split (301 clips).

Usage:
    python vlm_multiturn_var.py \\
        --data_dir ~/data/SoccerNet_Data \\
        --hdf5_dir ~/data/SoccerNet_HDF5_compact \\
        --vlm_model qwen3-vl-30b-a3b \\
        --num_frames 8 \\
        --output_path results/multiturn_var_test.json

    # Smoke test (first 20 actions)
    python vlm_multiturn_var.py \\
        --data_dir ~/data/SoccerNet_Data \\
        --hdf5_dir ~/data/SoccerNet_HDF5_compact \\
        --max_samples 20
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

import h5py
import numpy as np
import torch
from PIL import Image

# ── Label mappings (match SoccerNet-MVFoul evaluator) ────────────────────────

ACTION_CLASSES = [
    "Tackling", "Standing tackling", "High leg", "Holding",
    "Pushing", "Elbowing", "Challenge", "Dive",
]
SEVERITY_CLASSES = ["No offence", "No card", "Yellow card", "Red card"]

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}
SEVERITY_TO_IDX = {s: i for i, s in enumerate(SEVERITY_CLASSES)}

OFFENCE_SEV_MAP = {
    ("No offence", ""):  0,
    ("No Offence", ""):  0,
    ("Offence",   "1.0"): 1,
    ("Offence",   "3.0"): 2,
    ("Offence",   "5.0"): 3,
}

# Prompt response keys (lower) → canonical class
_ACTION_MAP = {
    "standing tackle":    "Standing tackling",
    "standing tackling":  "Standing tackling",
    "tackle":             "Tackling",
    "tackling":           "Tackling",
    "challenge":          "Challenge",
    "holding":            "Holding",
    "elbowing":           "Elbowing",
    "high leg":           "High leg",
    "pushing":            "Pushing",
    "dive":               "Dive",
    "diving":             "Dive",
}
_SEVERITY_MAP = {
    "no foul":     0,
    "no offence":  0,
    "no offense":  0,
    "minor foul":  1,
    "no card":     1,
    "yellow card": 2,
    "yellow":      2,
    "red card":    3,
    "red":         3,
}

# ── VLM registry ──────────────────────────────────────────────────────────────

VLM_REGISTRY = {
    # MoE 30B (3B activated) — fits on 1 GPU, 10× faster than dense 30B
    "qwen3-vl-30b-a3b": {
        "hf_id":  "/net/obelix/homes/aszostek/hf_models/qwen3-vl-30b",
        "family": "qwen3",
        "gpus":   1,  # 30B weights still ~60 GB
    },
    "qwen3-vl-30b": {
        "hf_id":  "Qwen/Qwen3-VL-30B-Instruct",
        "family": "qwen3",
        "gpus":   2,
    },
    "qwen3-vl-8b": {
        "hf_id":  "Qwen/Qwen3-VL-8B-Instruct",
        "family": "qwen3",
        "gpus":   1,
    },
    "qwen2.5-vl-7b": {
        "hf_id":  "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "qwen2",
        "gpus":   1,
    },
}

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert football VAR (Video Assistant Referee) analyst. "
    "You will examine a foul incident across multiple camera angles, "
    "one view at a time, systematically building your evidence before ruling. "
    "In each intermediate turn, describe only what you observe — no ruling yet. "
    "Give your classification only in the final turn when asked."
)

_VIEW_LABELS = {0: "Live Camera", 1: "Replay 1", 2: "Replay 2", 3: "Replay 3"}

_TURN_INSTRUCTIONS = {
    0: (
        "Look carefully at:\n"
        "• Approach angle and distance\n"
        "• Both players' positions and body posture\n"
        "• Whether the challenging player intended to play the ball\n"
        "• Any initial contact visible in this wide view\n\n"
        "Describe only what you see. Do NOT classify yet."
    ),
    1: (
        "This replay may show the contact more clearly. Examine:\n"
        "• The exact body part that made contact\n"
        "• The force and speed of the challenge\n"
        "• Whether the foot/leg was raised dangerously\n"
        "• The opponent's reaction and resulting position\n\n"
        "Add new evidence from this angle. Do NOT classify yet."
    ),
    2: (
        "This perpendicular angle helps judge depth and fall distance. Note:\n"
        "• Does the contact match what you described from other views?\n"
        "• Can you now determine the exact contact region (upper/lower body)?\n"
        "• Any indication of deception or simulation?\n\n"
        "Confirm or revise your observations. Do NOT classify yet."
    ),
}
_DEFAULT_TURN_INSTRUCTION = (
    "Observe this additional angle. Note any new evidence. Do NOT classify yet."
)

FINAL_PROMPT = (
    "You have now examined all available camera angles.\n\n"
    "Based on the accumulated evidence from all views, give your final ruling.\n\n"
    "Action (choose exactly one):\n"
    "  Standing Tackle / Tackle / Challenge / Holding / Elbowing / High Leg / Pushing / Dive\n\n"
    "Severity (choose exactly one):\n"
    "  No Foul / Minor Foul / Yellow Card / Red Card\n\n"
    "Where:\n"
    "  No Foul      = no offence committed (includes simulation/dive)\n"
    "  Minor Foul   = foul but no card warranted\n"
    "  Yellow Card  = reckless challenge / dissent / persistent fouling\n"
    "  Red Card     = excessive force or violent conduct endangering safety\n\n"
    "Reason briefly from the contact type, force, and danger level, then output exactly:\n"
    "ACTION: <class>\n"
    "SEVERITY: <class>"
)


def _build_user_content(
    frames: List[Image.Image],
    view_idx: int,
    n_views: int,
    thinking: bool,
) -> list:
    """Build the user content list for one view turn."""
    view_label = _VIEW_LABELS.get(view_idx, f"Replay {view_idx}")
    instruction = _TURN_INSTRUCTIONS.get(view_idx, _DEFAULT_TURN_INSTRUCTION)
    text = (
        f"[View {view_idx + 1} of {n_views}: {view_label}]\n\n"
        f"{instruction}"
    )
    # Prepend /think for Qwen3 thinking mode in the first user turn
    if thinking and view_idx == 0:
        text = "/think\n" + text

    content: list = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": text})
    return content


# ── VLM loading ───────────────────────────────────────────────────────────────


def _load_vlm(model_key: str, quantize: str = "", hf_cache: str = ""):
    if hf_cache:
        os.environ["HF_HOME"] = hf_cache
        os.environ["TRANSFORMERS_CACHE"] = hf_cache

    entry = VLM_REGISTRY[model_key]
    hf_id = entry["hf_id"]
    family = entry["family"]

    model_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantize in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        if quantize == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

    if family in ("qwen3", "qwen2"):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
			hf_id,
			torch_dtype=torch.bfloat16,
			device_map="auto",
		)
    else:
        raise ValueError(f"Unknown VLM family: {family!r}")

    model.eval()
    print(f"[vlm_multiturn_var] Loaded {hf_id} on {next(model.parameters()).device}")
    return model, processor, family


# ── Inference ─────────────────────────────────────────────────────────────────


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _generate_turn(
    model,
    processor,
    messages: list,
    thinking: bool,
    max_new_tokens: int = 512,
) -> str:
    """Run one forward pass over the full message history and return the response."""
    device = next(model.parameters()).device

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=thinking,
        ).to(device)
    except TypeError:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return _strip_think(raw) if thinking else raw


def run_multiturn(
    model,
    processor,
    frames_per_view: List[List[Image.Image]],
    thinking: bool,
    tokens_per_view: int = 768,
    tokens_final: int = 1024,
) -> Tuple[str, List[dict]]:
    """
    Run the structured multi-turn conversation for one action.

    Returns (final_response, messages_log).
    """
    n_views = len(frames_per_view)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for v_idx, frames in enumerate(frames_per_view):
        content = _build_user_content(frames, v_idx, n_views, thinking)
        messages.append({"role": "user", "content": content})

        resp = _generate_turn(model, processor, messages, thinking, tokens_per_view)
        messages.append({"role": "assistant", "content": resp})

    # Final classification turn — no images, just the ruling request
    messages.append({"role": "user", "content": [{"type": "text", "text": FINAL_PROMPT}]})
    final_resp = _generate_turn(model, processor, messages, thinking, tokens_final)
    messages.append({"role": "assistant", "content": final_resp})

    return final_resp, messages


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_annotations(data_dir: str, split: str) -> Dict[str, dict]:
    path = os.path.join(data_dir, split, "annotations.json")
    with open(path) as f:
        raw = json.load(f)
    samples = {}
    for aid, adat in raw["Actions"].items():
        action_class   = adat.get("Action class", "")
        offence_class  = adat.get("Offence", "")
        severity_class = adat.get("Severity", "")

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
        sev_idx = OFFENCE_SEV_MAP.get(key)
        if sev_idx is None and offence_class in ("No offence", "No Offence"):
            sev_idx = 0
        if sev_idx is None:
            continue

        act_idx = ACTION_TO_IDX.get(action_class, -1)
        if act_idx == -1:
            continue

        clips = [c["Url"].split("/")[-1] for c in adat.get("Clips", [])]
        samples[aid] = {"action": act_idx, "severity": sev_idx, "clips": clips}
    return samples


def _load_views(
    hdf5_file,
    action_id: str,
    clips: List[str],
    n_frames: int = 8,
    max_views: int = 4,
) -> List[List[Image.Image]]:
    """Return list of PIL-frame lists, one per available camera view."""
    views = []
    for clip_name in clips[:max_views]:
        clip_key = clip_name.replace(".mp4", "")
        hdf5_key = f"action_{action_id}/{clip_key}"
        if hdf5_key not in hdf5_file:
            continue
        frames_np = hdf5_file[hdf5_key][:]
        T = frames_np.shape[0]
        if T == 0:
            continue
        if T <= n_frames:
            indices = np.arange(T)
        else:
            # Centre-biased sampling: 2 early, dense contact window, 2 late
            center = T // 2
            lo = max(0, center - T // 5)
            hi = min(T - 1, center + T // 5)
            n_dense = max(1, n_frames - 4)
            n_edge  = (n_frames - n_dense) // 2
            early   = np.linspace(0,  lo, n_edge + 1, dtype=int)[:-1]
            dense   = np.linspace(lo, hi, n_dense,     dtype=int)
            late    = np.linspace(hi, T - 1, n_edge + 1, dtype=int)[1:]
            indices = np.unique(np.concatenate([early, dense, late]))[:n_frames]

        views.append([Image.fromarray(frames_np[i]) for i in sorted(indices)])
    return views


# ── Parsing & metrics ─────────────────────────────────────────────────────────


def _parse_response(text: str) -> Tuple[int, int]:
    """Parse ACTION: and SEVERITY: lines; return (act_idx, sev_idx) or (-1,-1)."""
    act_idx = sev_idx = -1

    m = re.search(r"ACTION:\s*(.+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().lower().rstrip(".,;")
        # Longest-match first to avoid "tackle" stealing "standing tackle"
        for label in sorted(_ACTION_MAP, key=len, reverse=True):
            if label in raw:
                act_idx = ACTION_TO_IDX[_ACTION_MAP[label]]
                break

    m = re.search(r"SEVERITY:\s*(.+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().lower().rstrip(".,;")
        for label in sorted(_SEVERITY_MAP, key=len, reverse=True):
            if label in raw:
                sev_idx = _SEVERITY_MAP[label]
                break

    return act_idx, sev_idx


def _balanced_acc(gt: List[int], pred: List[int], n_classes: int) -> float:
    if not gt:
        return 0.0
    per_class = []
    for c in range(n_classes):
        mask = [i for i, g in enumerate(gt) if g == c]
        if not mask:
            continue
        per_class.append(sum(1 for i in mask if pred[i] == c) / len(mask))
    return float(np.mean(per_class)) if per_class else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Turn VAR: structured cross-view reasoning (test set evaluation)"
    )
    parser.add_argument("--vlm_model",   default="qwen3-vl-30b-a3b",
                        choices=list(VLM_REGISTRY))
    parser.add_argument("--data_dir",    required=True,
                        help="SoccerNet_Data root: contains {split}/annotations.json")
    parser.add_argument("--hdf5_dir",   default=None,
                        help="Dir with {split}.hdf5 frames (e.g. SoccerNet_HDF5_compact). "
                             "Defaults to data_dir.")
    parser.add_argument("--split",       default="Test",
                        help="Dataset split to evaluate (default: Test)")
    parser.add_argument("--thinking",    action="store_true", default=True,
                        help="Use Qwen3 thinking mode (default: on)")
    parser.add_argument("--no_thinking", dest="thinking", action="store_false")
    parser.add_argument("--num_frames",  type=int, default=8,
                        help="Frames sampled per view (centre-biased)")
    parser.add_argument("--max_views",   type=int, default=4,
                        help="Maximum camera views per action")
    parser.add_argument("--tokens_per_view", type=int, default=768,
                        help="max_new_tokens for each evidence turn")
    parser.add_argument("--tokens_final",    type=int, default=512,
                        help="max_new_tokens for the final classification turn")
    parser.add_argument("--output_path", default="results/multiturn_var_test.json")
    parser.add_argument("--hf_cache_dir", default="")
    parser.add_argument("--quantize",   default="", choices=["", "4bit", "8bit"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap samples for smoke testing")
    args = parser.parse_args()

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    hdf5_dir = args.hdf5_dir or args.data_dir

    # Resume: load existing results so we can skip already-done actions
    existing: dict = {}
    if Path(args.output_path).exists():
        try:
            with open(args.output_path) as f:
                saved = json.load(f)
            for rec in saved.get("per_action", []):
                existing[rec["action_id"]] = rec
            print(f"[resume] Found {len(existing)} previously processed actions")
        except Exception:
            pass

    print(f"[vlm_multiturn_var] Model   : {args.vlm_model}")
    print(f"[vlm_multiturn_var] Split   : {args.split}")
    print(f"[vlm_multiturn_var] Thinking: {args.thinking}")

    model, processor, _ = _load_vlm(args.vlm_model, args.quantize, args.hf_cache_dir)

    hdf5_path = os.path.join(hdf5_dir, f"{args.split}.hdf5")
    if not os.path.exists(hdf5_path):
        sys.exit(f"ERROR: HDF5 not found: {hdf5_path}")

    annotations = _load_annotations(args.data_dir, args.split)
    action_ids = list(annotations.keys())
    if args.max_samples:
        action_ids = action_ids[: args.max_samples]

    print(f"[vlm_multiturn_var] Total annotated actions: {len(action_ids)}")

    records: List[dict] = list(existing.values())
    gt_act,  pred_act  = [], []
    gt_sev,  pred_sev  = [], []

    # Seed from existing results
    for rec in records:
        if rec.get("gt_action") is not None and rec.get("pred_action", -1) != -1:
            gt_act.append(rec["gt_action"]);  pred_act.append(rec["pred_action"])
        if rec.get("gt_severity") is not None and rec.get("pred_severity", -1) != -1:
            gt_sev.append(rec["gt_severity"]); pred_sev.append(rec["pred_severity"])

    with h5py.File(hdf5_path, "r") as hdf5:
        for i, aid in enumerate(action_ids):
            if aid in existing:
                continue

            ann = annotations[aid]
            gt_a, gt_s = ann["action"], ann["severity"]

            views = _load_views(
                hdf5, aid, ann["clips"], args.num_frames, args.max_views
            )
            if not views:
                print(f"  [SKIP] {aid}: no clips in HDF5")
                records.append({
                    "action_id": aid, "gt_action": gt_a, "gt_severity": gt_s,
                    "pred_action": -1, "pred_severity": -1,
                    "n_views": 0, "raw_output": "", "turn_log": [],
                })
                continue

            t0 = time.time()
            try:
                final_resp, msg_log = run_multiturn(
                    model, processor, views, args.thinking,
                    args.tokens_per_view, args.tokens_final,
                )
                pa, ps = _parse_response(final_resp)
                # Compact turn log: keep only text parts
                turn_summary = []
                for m in msg_log:
                    if m["role"] in ("user", "assistant") and isinstance(m["content"], list):
                        text_parts = [c["text"] for c in m["content"] if c.get("type") == "text"]
                        turn_summary.append({"role": m["role"], "text": " ".join(text_parts)})
                    elif m["role"] == "assistant" and isinstance(m["content"], str):
                        turn_summary.append({"role": "assistant", "text": m["content"]})
            except Exception as exc:
                print(f"  [ERROR] {aid}: {exc}")
                pa, ps = -1, -1
                final_resp = f"ERROR: {exc}"
                turn_summary = []

            elapsed = time.time() - t0
            rec = {
                "action_id":    aid,
                "gt_action":    gt_a,
                "gt_severity":  gt_s,
                "pred_action":  pa,
                "pred_severity": ps,
                "n_views":      len(views),
                "elapsed_s":    round(elapsed, 1),
                "final_response": final_resp,
                "turn_log":     turn_summary,
            }
            records.append(rec)

            if pa != -1:
                gt_act.append(gt_a);  pred_act.append(pa)
            if ps != -1:
                gt_sev.append(gt_s);  pred_sev.append(ps)

            # Progress log every 10 actions
            if (i + 1) % 10 == 0 or i == 0:
                sev_ba = _balanced_acc(gt_sev, pred_sev, 4) if gt_sev else 0.0
                act_ba = _balanced_acc(gt_act, pred_act, 8) if gt_act else 0.0
                print(
                    f"  [{i + 1:3d}/{len(action_ids)}]"
                    f"  views={len(views)}"
                    f"  BA_act={act_ba:.3f}  BA_sev={sev_ba:.3f}"
                    f"  parsed={len(gt_act)}/{len(records)}"
                    f"  {elapsed:.0f}s"
                )

            # Checkpoint after every action
            _save(args.output_path, records, gt_act, pred_act, gt_sev, pred_sev, args)

    act_bacc = _balanced_acc(gt_act, pred_act, 8)
    sev_bacc = _balanced_acc(gt_sev, pred_sev, 4)
    parse_rate = len(gt_act) / max(len(records), 1)

    summary = {
        "model":                      args.vlm_model,
        "split":                      args.split,
        "thinking":                   args.thinking,
        "num_frames":                 args.num_frames,
        "max_views":                  args.max_views,
        "n_evaluated":                len(records),
        "parse_rate":                 round(parse_rate, 4),
        "action_balanced_accuracy":   round(act_bacc, 4),
        "severity_balanced_accuracy": round(sev_bacc, 4),
        "leaderboard_value":          round((act_bacc + sev_bacc) / 2, 4),
    }

    print("\n=== Multi-Turn VAR Results ===")
    for k, v in summary.items():
        print(f"  {k:35s}: {v}")

    _save(args.output_path, records, gt_act, pred_act, gt_sev, pred_sev, args)
    print(f"\nSaved → {args.output_path}")


def _save(path, records, gt_act, pred_act, gt_sev, pred_sev, args):
    act_bacc = _balanced_acc(gt_act, pred_act, 8)
    sev_bacc = _balanced_acc(gt_sev, pred_sev, 4)
    payload = {
        "summary": {
            "model": args.vlm_model,
            "split": args.split,
            "thinking": args.thinking,
            "n_evaluated": len(records),
            "parse_rate": round(len(gt_act) / max(len(records), 1), 4),
            "action_balanced_accuracy":   round(act_bacc, 4),
            "severity_balanced_accuracy": round(sev_bacc, 4),
            "leaderboard_value":          round((act_bacc + sev_bacc) / 2, 4),
        },
        "per_action": records,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
