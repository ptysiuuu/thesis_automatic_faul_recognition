"""
vlm_zeroshot.py
===============
Zero-shot VLM evaluation on SoccerNet-MVFoul.

Loads a VLM and evaluates it directly on a dataset split without any
training, using a structured thinking-mode prompt that asks for action
class and severity. Computes balanced accuracy and saves per-sample results.

Usage:
    python vlm_zeroshot.py \
        --vlm_model qwen3-vl-30b \
        --data_dir ~/data/SoccerNet_HDF5 \
        --split Test \
        --thinking \
        --num_frames 8 \
        --output_path results/vlm_zeroshot_qwen3_30b.json \
        --hf_cache_dir ~/.cache/huggingface
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Label mappings ────────────────────────────────────────────────────────────

# Prompt label (lower) → canonical dataset name
_ACTION_PROMPT_MAP = {
    "standing tackle": "Standing tackling",
    "tackle":          "Tackling",
    "challenge":       "Challenge",
    "holding":         "Holding",
    "elbowing":        "Elbowing",
    "high leg":        "High leg",
    "pushing":         "Pushing",
    "dive":            "Dive",
}

_SEVERITY_PROMPT_MAP = {
    "no foul":     0,
    "minor foul":  1,
    "yellow card": 2,
    "red card":    3,
}

_ACTION_TO_IDX = {
    "Tackling": 0, "Standing tackling": 1, "High leg": 2, "Holding": 3,
    "Pushing": 4,  "Elbowing": 5,          "Challenge": 6, "Dive": 7,
}

_OFFENCE_SEV_MAP = {
    ("No offence", ""):  0,
    ("No Offence", ""):  0,
    ("Offence", "1.0"): 1,
    ("Offence", "3.0"): 2,
    ("Offence", "5.0"): 3,
}

# ── VLM registry ─────────────────────────────────────────────────────────────

VLM_REGISTRY = {
    "qwen2.5-vl-7b": {"hf_id": "Qwen/Qwen2.5-VL-7B-Instruct", "family": "qwen2"},
    "qwen3-vl-8b":   {"hf_id": "Qwen/Qwen3-VL-8B-Instruct",   "family": "qwen3"},
    "qwen3-vl-30b":  {"hf_id": "Qwen/Qwen3-VL-30B-Instruct",  "family": "qwen3"},
    "qwen3-vl-235b": {"hf_id": "Qwen/Qwen3-VL-235B-Instruct", "family": "qwen3"},
    "gemma4-12b":    {"hf_id": "google/gemma-4-12b-it",        "family": "gemma4"},
    "gemma4-31b":    {"hf_id": "google/gemma-4-31b-it",        "family": "gemma4"},
}

# ── Zero-shot prompt ──────────────────────────────────────────────────────────

TASK_PROMPT = (
    "You are a professional football referee reviewing a VAR clip.\n"
    "Identify the foul type and severity from these frames.\n\n"
    "Action: [Standing Tackle / Tackle / Challenge / Holding / "
    "Elbowing / High Leg / Pushing / Dive]\n"
    "Severity: [No Foul / Minor Foul / Yellow Card / Red Card]\n\n"
    "Reason step by step about: approach speed, contact type, body part hit, "
    "force and danger. Then output exactly:\n"
    "ACTION: <class>\n"
    "SEVERITY: <class>"
)

# ── VLM loading & inference ───────────────────────────────────────────────────

def _load_vlm(hf_id: str, family: str, quantize: str = ""):
    import torch
    from transformers import AutoProcessor

    dtype = torch.bfloat16
    model_kwargs: dict = {"torch_dtype": dtype, "device_map": "auto"}

    if quantize in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        if quantize == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

    if family == "qwen3":
        try:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(hf_id, **model_kwargs)
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(hf_id, **model_kwargs)
    elif family == "qwen2":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(hf_id, **model_kwargs)
    else:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(hf_id, **model_kwargs)

    model.eval()
    return model, processor


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _run_inference(model, processor, family: str, pil_frames, thinking: bool) -> str:
    import torch

    text = "/think\n" + TASK_PROMPT if (thinking and family == "qwen3") else TASK_PROMPT

    if family in ("qwen2", "qwen3"):
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": img} for img in pil_frames]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]

        apply_kw = {"tokenize": False, "add_generation_prompt": True}
        if thinking and family == "qwen3":
            try:
                apply_kw["enable_thinking"] = True
            except Exception:
                pass

        text_in = processor.apply_chat_template(messages, **apply_kw)
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(
            text=[text_in], images=img_in, videos=vid_in,
            padding=True, return_tensors="pt",
        ).to(model.device)

        max_new_tokens = 4096 if thinking else 512
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = out[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _strip_think(raw)

    else:
        # Gemma4 / generic
        content = [{"type": "image"} for _ in pil_frames]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_in], images=pil_frames,
            return_tensors="pt", padding=True,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024 if thinking else 512)
        generated = out[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _strip_think(raw)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_annotations(data_dir: str, split: str) -> Dict[str, dict]:
    ann_path = os.path.join(data_dir, split, "annotations.json")
    with open(ann_path) as f:
        return json.load(f)["Actions"]


def _gt_labels(action_data: dict) -> Tuple[Optional[int], Optional[int]]:
    action_class  = action_data.get("Action class", "")
    offence_class = action_data.get("Offence", "")
    severity_class = action_data.get("Severity", "")

    if action_class in ("", "Dont know"):
        return None, None
    if offence_class in ("", "Between") and action_class != "Dive":
        return None, None

    action_idx = _ACTION_TO_IDX.get(action_class)
    if action_idx is None:
        return None, None

    if offence_class in ("", "Between"):
        offence_class = "Offence"
    sev_idx = _OFFENCE_SEV_MAP.get((offence_class, severity_class))
    if sev_idx is None:
        return None, None

    return action_idx, sev_idx


def _load_frames(hdf5_file, action_id: str, num_frames: int) -> Optional[np.ndarray]:
    """Return [num_frames, H, W, C] uint8 by uniform sampling from clip_0."""
    key = f"{action_id}/clip_0"
    if key not in hdf5_file:
        return None
    frames = hdf5_file[key][:]  # [T, H, W, C] uint8
    T = frames.shape[0]
    if T == 0:
        return None
    indices = np.linspace(0, T - 1, num_frames, dtype=int)
    return frames[indices]


# ── Parsing & metrics ─────────────────────────────────────────────────────────

def _parse_response(text: str) -> Tuple[Optional[int], Optional[int]]:
    action_idx = sev_idx = None

    m = re.search(r"ACTION:\s*(.+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().lower()
        # longest match first to avoid "tackle" eating "standing tackle"
        for label in sorted(_ACTION_PROMPT_MAP, key=len, reverse=True):
            if label in raw:
                action_idx = _ACTION_TO_IDX[_ACTION_PROMPT_MAP[label]]
                break

    m = re.search(r"SEVERITY:\s*(.+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().lower()
        for label in sorted(_SEVERITY_PROMPT_MAP, key=len, reverse=True):
            if label in raw:
                sev_idx = _SEVERITY_PROMPT_MAP[label]
                break

    return action_idx, sev_idx


def _balanced_acc(gt: List[int], pred: List[int], n_classes: int) -> float:
    per_class = []
    for c in range(n_classes):
        mask = [i for i, g in enumerate(gt) if g == c]
        if not mask:
            continue
        per_class.append(sum(1 for i in mask if pred[i] == c) / len(mask))
    return float(np.mean(per_class)) if per_class else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model",   default="qwen3-vl-30b", choices=list(VLM_REGISTRY))
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--split",       default="Test")
    parser.add_argument("--thinking",    action="store_true")
    parser.add_argument("--num_frames",  type=int, default=8)
    parser.add_argument("--output_path", default="vlm_zeroshot_results.json")
    parser.add_argument("--hf_cache_dir", default="")
    parser.add_argument("--quantize",    default="", choices=["", "4bit", "8bit"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of samples (for debugging)")
    args = parser.parse_args()

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir

    reg = VLM_REGISTRY[args.vlm_model]
    print(f"[vlm_zeroshot] Model : {args.vlm_model} ({reg['hf_id']})")
    print(f"[vlm_zeroshot] Split : {args.split}  Thinking: {args.thinking}  Frames: {args.num_frames}")

    model, processor = _load_vlm(reg["hf_id"], reg["family"], args.quantize)

    import h5py
    from PIL import Image

    hdf5_path = os.path.join(args.data_dir, f"{args.split}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"ERROR: HDF5 file not found: {hdf5_path}", file=sys.stderr)
        sys.exit(1)

    annotations = _load_annotations(args.data_dir, args.split)

    records: List[dict] = []
    gt_act,  pred_act  = [], []
    gt_sev,  pred_sev  = [], []

    action_ids = list(annotations.keys())
    if args.max_samples:
        action_ids = action_ids[: args.max_samples]

    print(f"[vlm_zeroshot] Evaluating {len(action_ids)} actions...")

    with h5py.File(hdf5_path, "r") as hdf5:
        for i, aid in enumerate(action_ids):
            gt_a, gt_s = _gt_labels(annotations[aid])
            if gt_a is None:
                continue

            frames_np = _load_frames(hdf5, aid, args.num_frames)
            if frames_np is None:
                print(f"  [SKIP] {aid}: no clip_0 in HDF5")
                continue

            pil_frames = [Image.fromarray(f) for f in frames_np]

            raw = _run_inference(model, processor, reg["family"], pil_frames, args.thinking)
            pa, ps = _parse_response(raw)

            records.append({
                "action_id": aid,
                "gt_action": gt_a, "gt_severity": gt_s,
                "pred_action": pa,  "pred_severity": ps,
                "raw_output": raw,
            })

            if pa is not None:
                gt_act.append(gt_a);  pred_act.append(pa)
            if ps is not None:
                gt_sev.append(gt_s);  pred_sev.append(ps)

            if (i + 1) % 50 == 0:
                sev_ba = _balanced_acc(gt_sev, pred_sev, 4) if gt_sev else 0.0
                print(f"  [{i+1}/{len(action_ids)}]  sev_bacc={sev_ba:.3f}  "
                      f"parsed={len(gt_act)}/{len(records)}")

    act_bacc  = _balanced_acc(gt_act, pred_act, 8)
    sev_bacc  = _balanced_acc(gt_sev, pred_sev, 4)
    parse_rate = len(gt_act) / max(len(records), 1)

    summary = {
        "model":                       args.vlm_model,
        "split":                       args.split,
        "thinking":                    args.thinking,
        "num_frames":                  args.num_frames,
        "n_evaluated":                 len(records),
        "parse_rate":                  round(parse_rate, 4),
        "action_balanced_accuracy":    round(act_bacc, 4),
        "severity_balanced_accuracy":  round(sev_bacc, 4),
    }

    print("\n=== Zero-shot Results ===")
    for k, v in summary.items():
        print(f"  {k:35s}: {v}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({"summary": summary, "per_action": records}, f, indent=2)
    print(f"\nSaved → {args.output_path}")


if __name__ == "__main__":
    main()
