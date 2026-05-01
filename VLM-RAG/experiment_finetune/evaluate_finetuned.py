"""
evaluate_finetuned.py
=====================
Evaluates the LoRA-finetuned VLM on the full Test set and produces
SoccerNet leaderboard predictions.json compatible with the official evaluator.

Compares four configurations:
  A) Base model, zero_shot
  B) Base model, rule_grounded
  C) Finetuned model, zero_shot        ← does finetuning help without rules?
  D) Finetuned model, rule_grounded    ← best expected configuration

Usage:
  python evaluate_finetuned.py \
    --hdf5_path     /net/tscratch/people/plgaszos/SoccerNet_HDF5/Test.hdf5 \
    --annotations   /net/tscratch/people/plgaszos/SoccerNet_Data/Test/annotations.json \
    --adapter_path  /net/tscratch/people/plgaszos/vlm_finetuned/lora_adapters \
    --law12_pdf     /net/tscratch/people/plgaszos/SoccerNet_Data/law12.pdf \
    --output_dir    vlm_test_results \
    --frames_per_view 4
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import h5py
import torch
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

from law12_rag import Law12RAG
from vlm_classifier import (
    build_prompt,
    parse_response,
    SYSTEM_PROMPT,
    ACTION_CLASSES,
    SEVERITY_CLASSES,
    ACTION_TO_IDX,
    SEVERITY_TO_IDX,
)
from evaluate_vlm import load_annotations, compute_metrics

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_UTILS = True
except ImportError:
    HAS_QWEN_UTILS = False


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def load_model(
    base_model_name: str,
    adapter_path: str = None,
) -> tuple:
    """
    Load Qwen2.5-VL with optional LoRA adapters.

    Returns (model, processor)
    """
    print(f"Loading base model: {base_model_name}")
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading LoRA adapters from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # merge for faster inference
        print("Adapters merged into base model.")
    else:
        print("No adapter path provided — using base model only.")

    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def classify_sample(
    model,
    processor,
    frames_per_view: list,  # List[List[PIL.Image]]
    strategy: str,
    rag: Law12RAG,
    action_hint: str = "Dont know",
    max_new_tokens: int = 256,
) -> tuple:
    """
    Classify one multi-view sample.
    Returns (action_idx, severity_idx, raw_response)
    """
    # RAG context
    if strategy != "zero_shot":
        query = rag.build_query(action_hint)
        law12_ctx = rag.retrieve(query)
    else:
        law12_ctx = ""

    prompt = build_prompt(
        strategy, n_views=len(frames_per_view), law12_context=law12_ctx
    )

    # Build message
    content = []
    for v_idx, frames in enumerate(frames_per_view):
        view_label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
        content.append({"type": "text", "text": f"\n[{view_label}]"})
        for frame in frames:
            content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": f"\n\n{prompt}"})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    if HAS_QWEN_UTILS:
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
    else:
        # Fallback without qwen_vl_utils
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0]

    action_idx, severity_idx = parse_response(raw)
    return action_idx, severity_idx, raw


# ---------------------------------------------------------------------------
# Decode to SoccerNet predictions.json format
# ---------------------------------------------------------------------------


def build_predictions_json(
    action_ids: list,
    pred_actions: list,
    pred_severities: list,
    set_name: str = "test",
) -> dict:
    """
    Build predictions.json compatible with SoccerNet official evaluator.
    """
    actions = {}
    severity_to_offence = {
        0: ("No offence", ""),
        1: ("Offence", "1.0"),
        2: ("Offence", "3.0"),
        3: ("Offence", "5.0"),
    }

    for action_id, pred_act, pred_sev in zip(action_ids, pred_actions, pred_severities):
        if pred_act == -1:
            pred_act = 0  # default to Tackling on parse failure
        if pred_sev == -1:
            pred_sev = 0  # default to No offence

        offence, severity = severity_to_offence[pred_sev]
        actions[action_id] = {
            "Action class": ACTION_CLASSES[pred_act],
            "Offence": offence,
            "Severity": severity,
        }

    return {"Set": set_name, "Actions": actions}


# ---------------------------------------------------------------------------
# Full evaluation loop
# ---------------------------------------------------------------------------


def evaluate_config(
    model,
    processor,
    hdf5_path: str,
    samples: dict,
    strategy: str,
    rag: Law12RAG,
    frames_per_view: int = 4,
    use_gt_hint: bool = False,  # True = use ground truth for RAG query
    # False = use "Dont know" (realistic)
) -> dict:
    y_true_action, y_pred_action = [], []
    y_true_severity, y_pred_severity = [], []
    predictions_raw = {}

    with h5py.File(hdf5_path, "r", swmr=True) as hdf5:
        for action_id, sample in tqdm(samples.items()):
            action_key = f"action_{action_id}"

            # Load frames
            frames_per_view_list = []
            for clip_key_raw in sample["clips"][:4]:
                clip_key = clip_key_raw.replace(".mp4", "")
                key = f"{action_key}/{clip_key}"
                if key not in hdf5:
                    continue
                frames_np = hdf5[key][:]
                T = len(frames_np)
                if T < 2:
                    continue
                indices = np.linspace(0, T - 1, frames_per_view, dtype=int)
                pil_frames = [Image.fromarray(frames_np[i]) for i in indices]
                frames_per_view_list.append(pil_frames)

            if not frames_per_view_list:
                y_true_action.append(sample["action"])
                y_pred_action.append(-1)
                y_true_severity.append(sample["severity"])
                y_pred_severity.append(-1)
                continue

            # Action hint for RAG
            if use_gt_hint:
                hint = ACTION_CLASSES[sample["action"]]
            else:
                hint = "Dont know"  # realistic: we don't know the action

            try:
                act_idx, sev_idx, raw = classify_sample(
                    model=model,
                    processor=processor,
                    frames_per_view=frames_per_view_list,
                    strategy=strategy,
                    rag=rag,
                    action_hint=hint,
                )
            except Exception as e:
                print(f"Error on {action_id}: {e}")
                act_idx, sev_idx, raw = -1, -1, str(e)

            y_true_action.append(sample["action"])
            y_pred_action.append(act_idx)
            y_true_severity.append(sample["severity"])
            y_pred_severity.append(sev_idx)
            predictions_raw[action_id] = {
                "pred_action": act_idx,
                "pred_severity": sev_idx,
                "true_action": sample["action"],
                "true_severity": sample["severity"],
                "raw": raw,
            }

    metrics = compute_metrics(
        y_true_action,
        y_pred_action,
        y_true_severity,
        y_pred_severity,
    )
    metrics["predictions_raw"] = predictions_raw

    # Also build SoccerNet-format predictions
    all_ids = list(samples.keys())
    metrics["predictions_soccernet"] = build_predictions_json(
        action_ids=all_ids,
        pred_actions=y_pred_action,
        pred_severities=y_pred_severity,
    )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    print("Loading annotations...")
    samples = load_annotations(args.annotations)
    print(f"Found {len(samples)} test samples.")

    # Initialize RAG
    rag = Law12RAG(pdf_path=args.law12_pdf, top_k=3, use_embeddings=True)

    # Define configurations to evaluate
    configs = [
        # (config_name, adapter_path, strategy, use_gt_hint)
        ("base_zero_shot", None, "zero_shot", False),
        ("base_rule_grounded", None, "rule_grounded", False),
        ("finetuned_zero_shot", args.adapter_path, "zero_shot", False),
        ("finetuned_rule_grounded", args.adapter_path, "rule_grounded", False),
        # Two-stage: use predicted action (from zero_shot pass) as RAG hint
        # Implemented below separately
    ]

    all_results = {}
    loaded_models = {}  # cache to avoid reloading same model twice

    for config_name, adapter_path, strategy, use_gt_hint in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")

        # Load model (cache by adapter_path)
        cache_key = str(adapter_path)
        if cache_key not in loaded_models:
            model, processor = load_model(args.base_model, adapter_path)
            loaded_models[cache_key] = (model, processor)
        else:
            model, processor = loaded_models[cache_key]

        metrics = evaluate_config(
            model=model,
            processor=processor,
            hdf5_path=args.hdf5_path,
            samples=samples,
            strategy=strategy,
            rag=rag,
            frames_per_view=args.frames_per_view,
            use_gt_hint=use_gt_hint,
        )
        all_results[config_name] = metrics

        # Save SoccerNet predictions.json for official evaluator
        pred_path = output_dir / f"{config_name}_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(metrics["predictions_soccernet"], f)
        print(f"  Predictions saved to {pred_path}")
        print(
            f"  LB: {metrics.get('leaderboard_value', 0):.4f} | "
            f"Act BA: {metrics.get('balanced_acc_action', 0):.2f} | "
            f"Sev BA: {metrics.get('balanced_acc_severity', 0):.2f}"
        )

    # Comparison table
    print(f"\n{'='*75}")
    print("FINAL COMPARISON")
    print(f"{'Config':<30} {'Parse%':>7} {'Act BA':>8} {'Sev BA':>8} {'LB Val':>8}")
    print("-" * 65)
    for name, m in all_results.items():
        print(
            f"{name:<30} {m.get('parse_rate',0):>7.1f} "
            f"{m.get('balanced_acc_action',0):>8.2f} "
            f"{m.get('balanced_acc_severity',0):>8.2f} "
            f"{m.get('leaderboard_value',0):>8.4f}"
        )

    # Save summary
    summary = {
        k: {
            kk: vv
            for kk, vv in v.items()
            if kk
            not in (
                "predictions_raw",
                "predictions_soccernet",
                "confusion_action",
                "confusion_severity",
            )
        }
        for k, v in all_results.items()
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir}/summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="Path to LoRA adapters. If None, only base model configs run.",
    )
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--law12_pdf", default=None)
    parser.add_argument("--frames_per_view", type=int, default=4)
    parser.add_argument("--output_dir", default="vlm_test_results")
    args = parser.parse_args()
    main(args)
