"""
prepare_vlm_dataset.py
======================
Converts SoccerNet-MVFoul HDF5 + annotations into a format suitable
for VLM finetuning via LoRA.

Output format: JSONL where each line is one training sample:
{
  "id": "action_42",
  "images": ["path/to/frame_v0_t0.jpg", ...],  # saved to disk
  "conversations": [
    {"role": "user",    "content": "<image>...<image> [prompt]"},
    {"role": "assistant","content": "{\"action\": \"Tackling\", \"severity\": \"Red card\"}"}
  ]
}

This format is compatible with:
  - LLaMA-Factory (recommended)
  - Swift (modelscope)
  - TRL SFTTrainer with custom collator

Usage:
  python prepare_vlm_dataset.py \
    --hdf5_root  /net/tscratch/people/plgaszos/SoccerNet_HDF5 \
    --data_root  /net/tscratch/people/plgaszos/SoccerNet_Data \
    --output_dir /net/tscratch/people/plgaszos/vlm_dataset \
    --frames_per_view 4 \
    --strategy rule_grounded \
    --law12_pdf  /net/tscratch/people/plgaszos/SoccerNet_Data/law12.pdf
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import h5py
from tqdm import tqdm

# Reuse from your existing pipeline
from law12_rag import Law12RAG
from vlm_classifier import (
    build_prompt,
    ACTION_CLASSES,
    SEVERITY_CLASSES,
    ACTION_TO_IDX,
    SEVERITY_TO_IDX,
)

# ---------------------------------------------------------------------------
# Label loading (same filtering as your dataset.py)
# ---------------------------------------------------------------------------

OFFENCE_SEVERITY_MAP = {
    ("No offence", ""): 0,
    ("No Offence", ""): 0,
    ("Offence", "1.0"): 1,
    ("Offence", "3.0"): 2,
    ("Offence", "5.0"): 3,
}


def load_split_annotations(data_root: str, split: str) -> dict:
    path = os.path.join(data_root, split, "annotations.json")
    with open(path) as f:
        data = json.load(f)

    samples = {}
    for action_id, action_data in data["Actions"].items():
        action_class = action_data.get("Action class", "")
        offence_class = action_data.get("Offence", "")
        severity_class = action_data.get("Severity", "")

        if action_class in {"Dont know", ""}:
            continue
        if offence_class in {"Between", ""} and action_class != "Dive":
            continue
        if (
            severity_class in {"2.0", "4.0"}
            and action_class != "Dive"
            and offence_class not in ("No offence", "No Offence")
        ):
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
            "action": action_idx,
            "action_name": action_class,
            "severity": severity_idx,
            "severity_name": SEVERITY_CLASSES[severity_idx],
            "clips": clips,
        }
    return samples


# ---------------------------------------------------------------------------
# Frame extraction and saving
# ---------------------------------------------------------------------------


def extract_and_save_frames(
    hdf5: h5py.File,
    action_key: str,
    clip_key: str,
    output_dir: str,
    n_frames: int = 4,
) -> list:
    """
    Extract N evenly-spaced frames, save as JPEG, return paths.
    Returns [] if clip not found.
    """
    key = f"{action_key}/{clip_key}"
    if key not in hdf5:
        return []

    frames_np = hdf5[key][:]  # [T, H, W, C] uint8
    T = len(frames_np)
    if T < 2:
        return []

    indices = np.linspace(0, T - 1, n_frames, dtype=int)
    paths = []
    for i, idx in enumerate(indices):
        img = Image.fromarray(frames_np[idx])
        fname = f"{action_key}_{clip_key}_f{i:02d}.jpg"
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath):  # skip if already extracted
            img.save(fpath, quality=85)
        paths.append(fpath)
    return paths


# ---------------------------------------------------------------------------
# Prompt and answer construction
# ---------------------------------------------------------------------------


def build_training_sample(
    action_id: str,
    sample: dict,
    image_paths: list,  # flat list of all frame paths
    n_views: int,
    rag: Law12RAG,
    strategy: str,
    augment_answer: bool = True,
) -> dict:
    """
    Build one training sample in conversation format.

    augment_answer: if True, include a brief reasoning in the answer
                    (teaches the model WHY, not just WHAT)
    """
    # Build RAG context using ground truth action (we have labels during training)
    query = rag.build_query(sample["action_name"])
    law12_ctx = rag.retrieve(query)
    prompt_text = build_prompt(strategy, n_views=n_views, law12_context=law12_ctx)

    # Build user content: interleave view labels + <image> tags
    # The VLM processor replaces <image> tokens with actual image embeddings
    user_content = ""
    frame_idx = 0
    frames_per_view = len(image_paths) // n_views
    for v in range(n_views):
        view_label = "Live camera" if v == 0 else f"Replay {v}"
        user_content += f"\n[{view_label}]\n"
        for _ in range(frames_per_view):
            user_content += "<image>\n"
            frame_idx += 1

    user_content += f"\n{prompt_text}"

    # Build ground truth answer
    action_name = sample["action_name"]
    severity_name = sample["severity_name"]

    if augment_answer:
        # Include brief reasoning — teaches the model to cite rules
        # This is the key advantage over hard labels
        reasoning_map = {
            0: "No foul committed or simulation.",
            1: "Careless challenge — free kick, no card warranted.",
            2: "Reckless challenge — disregard for opponent's safety, yellow card.",
            3: "Excessive force or brutality — endangers opponent's safety, red card.",
        }
        reasoning = reasoning_map[sample["severity"]]
        answer = json.dumps(
            {
                "action": action_name,
                "severity": severity_name,
                "reasoning": reasoning,
            }
        )
    else:
        answer = json.dumps(
            {
                "action": action_name,
                "severity": severity_name,
            }
        )

    return {
        "id": f"action_{action_id}",
        "images": image_paths,
        "conversations": [
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": answer},
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Initialize RAG
    rag = Law12RAG(
        pdf_path=args.law12_pdf,
        top_k=3,
        use_embeddings=True,
    )

    splits_to_process = {
        "Train": "train.jsonl",
        "Valid": "valid.jsonl",
        "Test": "test.jsonl",
    }

    for split, out_fname in splits_to_process.items():
        hdf5_path = os.path.join(args.hdf5_root, f"{split}.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"Skipping {split} — HDF5 not found at {hdf5_path}")
            continue

        print(f"\nProcessing {split}...")
        samples = load_split_annotations(args.data_root, split)
        print(f"  {len(samples)} valid samples")

        out_path = output_dir / out_fname
        n_written = 0

        with h5py.File(hdf5_path, "r", swmr=True) as hdf5:
            with open(out_path, "w") as out_f:
                for action_id, sample in tqdm(samples.items(), desc=f"  {split}"):
                    action_key = f"action_{action_id}"

                    # Extract frames for each clip (up to 4 views)
                    all_image_paths = []
                    n_views_found = 0

                    for clip_key_raw in sample["clips"][:4]:
                        clip_key = clip_key_raw.replace(".mp4", "")
                        paths = extract_and_save_frames(
                            hdf5=hdf5,
                            action_key=action_key,
                            clip_key=clip_key,
                            output_dir=str(frames_dir),
                            n_frames=args.frames_per_view,
                        )
                        if paths:
                            all_image_paths.extend(paths)
                            n_views_found += 1

                    if n_views_found == 0:
                        continue

                    # Build training sample
                    record = build_training_sample(
                        action_id=action_id,
                        sample=sample,
                        image_paths=all_image_paths,
                        n_views=n_views_found,
                        rag=rag,
                        strategy=args.strategy,
                        augment_answer=(split == "Train"),
                    )
                    out_f.write(json.dumps(record) + "\n")
                    n_written += 1

        print(f"  Written: {n_written} samples → {out_path}")

    # Write dataset config for LLaMA-Factory
    dataset_config = {
        "vlm_train": {
            "file_name": str(output_dir / "train.jsonl"),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
        },
        "vlm_valid": {
            "file_name": str(output_dir / "valid.jsonl"),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
        },
    }
    config_path = output_dir / "dataset_info.json"
    with open(config_path, "w") as f:
        json.dump(dataset_config, f, indent=2)
    print(f"\nDataset config saved to {config_path}")
    print(f"Total frames saved to {frames_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--law12_pdf", default=None)
    parser.add_argument("--frames_per_view", type=int, default=4)
    parser.add_argument(
        "--strategy",
        default="rule_grounded",
        choices=["zero_shot", "rule_grounded", "chain_of_thought", "few_shot"],
    )
    args = parser.parse_args()
    main(args)
