#!/usr/bin/env python3
"""
extract_evidence.py
====================
Run VLM (Qwen2.5-VL-7B-Instruct) in video mode to extract structured
physical-evidence annotations for SoccerNet-MVFoul samples and save as JSONL.

Usage examples:
  # smoke test 50 samples
  python extract_evidence.py --hdf5_path /path/Valid.hdf5 --annotations /path/Valid/annotations.json \
      --output_dir /path/out --max_samples 50

  # full extraction (runs on all entries in annotations.json)
  python extract_evidence.py --hdf5_path /path/All.hdf5 --annotations /path/All/annotations.json \
      --output_dir /path/out

Use --video_mode to enable native video inputs for supported backends.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import h5py

# Add rag_icl root to path so vlm_pipeline is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_pipeline.utils.annotations import load_annotations
from vlm_pipeline.utils.frames import extract_all_views, extract_all_views_from_video
from vlm_pipeline.backends import get_backend
from vlm_pipeline.utils.constants import ACTION_CLASSES, SEVERITY_CLASSES

EVIDENCE_SCHEMA = {
    "contact_body_part": ["foot", "knee", "elbow", "shoulder", "head", "hand"],
    "foot_height_at_contact": ["ground", "ankle", "knee", "hip", "chest", "head"],
    "opponent_displacement": ["none", "stumble", "fall", "launched"],
    "challenging_player_speed": ["slow", "medium", "fast"],
    "ball_proximity": ["on_ball", "near", "far", "no_ball"],
    "contact_location_on_opponent": ["leg", "body", "arm", "head"],
    "player_balance_at_contact": ["controlled", "off_balance", "airborne"],
}


def build_prompt(n_views: int) -> str:
    return f"""You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

In 2-3 sentences, describe: which player initiates contact, what body part is used,
where on the opponent contact is made, and what happens to the opponent afterwards.
Be factual and specific about the physical action.

Respond with ONLY a valid JSON object in this exact format:
{{"vlm_description": "your 2-3 sentence description goes here"}}"""


def safe_parse_json(text: str):
    """Try to robustly extract and parse a JSON object from model output."""
    text = text.strip()
    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def infer_split_from_path(annotations_path: str) -> str:
    p = annotations_path.lower()
    if "train" in p:
        return "train"
    if "valid" in p or "val" in p:
        return "valid"
    return "test"


def main():
    parser = argparse.ArgumentParser(description="Extract VLM evidence annotations")
    parser.add_argument("--hdf5_path", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument(
        "--use_video_files",
        action="store_true",
        help="Load frames from dataset folders instead of HDF5",
    )
    parser.add_argument(
        "--video_root",
        default=None,
        help="Root folder of SoccerNet dataset (contains Train/Valid/Test)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="valid",
        help="Dataset split to process when hdf5/annotations are not provided",
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--video_mode",
        action="store_true",
        help="Enable native video inputs for supported backends.",
    )
    parser.add_argument("--output_dir", default="evidence_extraction")
    parser.add_argument("--frames_per_view", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    # Resolve paths based on split if explicit paths were not provided
    split = args.split
    if args.use_video_files:
        data_root = os.environ.get("DATA_ROOT", "./")
        split_cap = (
            "Train" if split == "train" else ("Valid" if split == "valid" else "Test")
        )
        if args.video_root is None:
            args.video_root = data_root
        if args.annotations is None:
            args.annotations = os.path.join(data_root, split_cap, "annotations.json")
    else:
        if args.annotations is None or args.hdf5_path is None:
            hdf5_root = os.environ.get("HDF5_ROOT", "./")
            data_root = os.environ.get("DATA_ROOT", "./")
            split_cap = (
                "Train"
                if split == "train"
                else ("Valid" if split == "valid" else "Test")
            )
            if args.hdf5_path is None:
                args.hdf5_path = os.path.join(hdf5_root, f"{split_cap}.hdf5")
            if args.annotations is None:
                args.annotations = os.path.join(
                    data_root, split_cap, "annotations.json"
                )

    samples = load_annotations(args.annotations)
    if args.max_samples:
        ids = list(samples.keys())[: args.max_samples]
        samples = {k: samples[k] for k in ids}

    # If annotations path name contains split information, prefer that
    split = infer_split_from_path(args.annotations) if args.annotations else split

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"evidence_{split}.jsonl"

    backend = get_backend(args.model_name, use_video_mode=args.video_mode)

    # Resume logic: collect already-processed action_ids from existing file
    processed = set()
    if out_file.exists():
        try:
            with open(out_file, "r") as fin:
                for line in fin:
                    try:
                        obj = json.loads(line)
                        aid = obj.get("action_id")
                        if aid is not None:
                            processed.add(str(aid))
                    except Exception:
                        continue
        except Exception:
            pass

    hdf5_handle = None
    if not args.use_video_files:
        hdf5_handle = h5py.File(args.hdf5_path, "r", swmr=True)

    try:
        with open(out_file, "a") as fout:
            for action_id, sample in samples.items():
                if str(action_id) in processed:
                    continue
                action_key = f"action_{action_id}"
                try:
                    if args.use_video_files:
                        fpv = extract_all_views_from_video(
                            args.video_root,
                            split,
                            action_id,
                            sample["clips"],
                            n_frames=args.frames_per_view,
                            weighted=False,
                            max_views=4,
                        )
                    else:
                        fpv = extract_all_views(
                            hdf5_handle,
                            action_key,
                            sample["clips"],
                            n_frames=args.frames_per_view,
                            weighted=False,
                            max_views=4,
                        )
                except Exception as e:
                    print(f"Skipping {action_id}: could not load frames: {e}")
                    continue

                if not fpv:
                    print(f"Skipping {action_id}: no frames")
                    continue
                prompt = build_prompt(n_views=len(fpv))

                try:
                    raw = backend.classify(fpv, prompt)
                except Exception as e:
                    print(f"Model error on {action_id}: {e}")
                    record = {
                        "action_id": action_id,
                        "split": split,
                        "vlm_model": args.model_name,
                        "error": str(e),
                        "true_action": sample.get("action"),
                        "true_severity": sample.get("severity"),
                    }
                    fout.write(json.dumps(record) + "\n")
                    continue

                parsed = safe_parse_json(raw)
                if parsed is None:
                    # fallback: save raw text with low confidence
                    record = {
                        "action_id": action_id,
                        "split": split,
                        "vlm_model": args.model_name,
                        "vlm_description": raw.strip()[:512],
                        "extraction_confidence": "low",
                        "true_action": sample.get("action"),
                        "true_severity": sample.get("severity"),
                    }
                    fout.write(json.dumps(record) + "\n")
                    print(f"Parse failed for {action_id}, saved raw.")
                    continue
                print(f"DEBUG {action_id}: {parsed}")

                # Build final record merging gold labels
                record = {
                    "action_id": action_id,
                    "split": split,
                    "vlm_model": args.model_name,
                    "true_action": sample.get("action"),
                    "true_severity": sample.get("severity"),
                    "vlm_description": (
                        parsed.get("vlm_description", "") if parsed else raw.strip()
                    ),
                    "extraction_confidence": "high" if parsed else "low",
                }

                fout.write(json.dumps(record) + "\n")

    finally:
        if hdf5_handle is not None:
            hdf5_handle.close()

    print(f"Wrote evidence file: {out_file}")


if __name__ == "__main__":
    main()
