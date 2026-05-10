"""
evaluate_local.py
=================
Runs VLM ablation experiments locally using:
  - Ollama for inference (no GPU needed, model hosted on your PC)
  - Raw SoccerNet mp4 files (no HDF5 cluster files needed)

This is a drop-in replacement for evaluate_ablation.py / evaluate_ablation_v2.py
for local experimentation when cluster compute is unavailable.

Prerequisites:
    1. ollama serve                          # start Ollama
    2. ollama pull qwen2.5vl:7b             # pull the model
    3. pip install requests opencv-python-headless pillow tqdm scikit-learn

SoccerNet directory structure expected:
    <data_root>/
    ├── Train/
    │   ├── annotations.json
    │   └── action_0/clip_0.mp4 ...
    ├── Valid/
    │   ├── annotations.json
    │   └── action_0/clip_0.mp4 ...
    └── law12.pdf   (optional, uses hardcoded fallback if missing)

Usage:
    # Smoke test — 10 samples, fastest strategy
    python evaluate_local.py \\
        --data_root ~/SoccerNet \\
        --strategy  static_few_shot \\
        --max_samples 10

    # cos_two_stage — best strategy, needs medoid cache from cluster
    python evaluate_local.py \\
        --data_root  ~/SoccerNet \\
        --strategy   cos_two_stage \\
        --medoid_cache ~/medoid_cache.json \\
        --max_samples 50

    # Custom Ollama host or model tag
    python evaluate_local.py \\
        --data_root ~/SoccerNet \\
        --strategy  static_few_shot \\
        --ollama_host http://localhost:11434 \\
        --ollama_model qwen2.5vl:7b-instruct-q4_K_M \\
        --max_samples 20

Notes:
    - Speed: ~15-30s per sample on a modern CPU (M-series Mac, Ryzen 9, etc.)
              ~5-8s per sample with a consumer GPU (RTX 3080+)
    - The medoid cache (medoid_cache.json or medoid_cache_v2.json) was built
      on the cluster and contains base64 JPEG frames — copy it locally.
    - The FAISS index is NOT needed for strategies 0, 1, 2, 4 (most useful ones).
      Only rag_icl (strategy 3) requires it.
    - Without --train_annotations, hardcoded severity priors are used.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from vlm_pipeline.utils.constants import (
    ACTION_CLASSES,
    SEVERITY_CLASSES,
    PER_ACTION_SEVERITY_PRIOR,
    SEVERITY_PRIOR_DEFAULT,
)
from vlm_pipeline.utils.annotations import compute_metrics
from vlm_pipeline.utils.frames import (
    parse_key_frames,
    select_key_frames,
    format_selected_frame_info,
)
from vlm_pipeline.utils.frames_local import (
    load_annotations_local,
    extract_all_views_local,
)
from vlm_pipeline.retrieval.law12_rag import Law12RAG
from vlm_pipeline.retrieval.medoid_cache import (
    load_medoid_cache,
    build_examples_text,
    build_severity_examples,
    build_targeted_examples,
)
from vlm_pipeline.prompts import (
    build_static_prompt,
    build_data_driven_prompt,
    build_two_stage_action_prompt,
    build_two_stage_severity_prompt,
    build_cos_frame_selection_prompt,
    build_cos_action_prompt,
    build_cos_severity_prompt,
)
from vlm_pipeline.strategies.base import (
    parse_response,
    parse_action_only,
    parse_severity_only,
)
from vlm_pipeline.strategies.description_first import DescriptionFirstStrategy
from vlm_pipeline.strategies.severity_focused import (
    run_cos_two_stage_description_severity,
    run_cos_static_sev,
)
from vlm_pipeline.backends.ollama import OllamaBackend

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Strategy:    {args.strategy}")
    print(f"Ollama model: {args.ollama_model}")
    print(f"Data root:   {args.data_root}")
    print(f"Split:       {args.split}")
    print(f"{'='*60}")

    # ── Load annotations ───────────────────────────────────────────────────
    samples = load_annotations_local(args.data_root, args.split)
    if args.max_samples:
        ids = list(samples.keys())[: args.max_samples]
        samples = {k: samples[k] for k in ids}
    print(f"Evaluating on {len(samples)} samples.")

    # ── Law 12 RAG ─────────────────────────────────────────────────────────
    law12_pdf = args.law12_pdf or os.path.join(args.data_root, "law12.pdf")
    rag = Law12RAG(
        pdf_path=law12_pdf if os.path.exists(law12_pdf) else None,
        top_k=3,
        use_embeddings=True,
    )

    # ── Severity priors ────────────────────────────────────────────────────
    severity_priors = SEVERITY_PRIOR_DEFAULT
    per_action_priors = PER_ACTION_SEVERITY_PRIOR

    if args.train_annotations and os.path.exists(args.train_annotations):
        from vlm_pipeline.utils.annotations import (
            compute_severity_priors,
            compute_per_action_severity_priors,
        )

        severity_priors = compute_severity_priors(args.train_annotations)
        per_action_priors = compute_per_action_severity_priors(args.train_annotations)
        print("[Priors] Computed from training annotations.")
    else:
        print("[Priors] Using hardcoded defaults (no --train_annotations provided).")

    # ── Medoid cache ───────────────────────────────────────────────────────
    medoid_cache = None
    needs_medoid = args.strategy in {
        "data_driven",
        "two_stage",
        "cos_two_stage",
        "cos_disambig",
        "cos_two_stage_description_severity",
        "cos_static_sev",
    }
    if needs_medoid:
        if not args.medoid_cache or not os.path.exists(args.medoid_cache):
            print(
                f"WARNING: strategy '{args.strategy}' needs a medoid cache.\n"
                "Copy medoid_cache.json (or medoid_cache_v2.json) from the cluster:\n"
                "  scp plgaszos@athena:/net/tscratch/people/plgaszos/vlm_rag_icl/medoid_cache_v2.json .\n"
                "Then pass: --medoid_cache ./medoid_cache_v2.json\n"
                "Falling back to static_few_shot."
            )
            args.strategy = "static_few_shot"
        else:
            medoid_cache = load_medoid_cache(args.medoid_cache)
            print(f"[MedoidCache] Loaded {len(medoid_cache)} entries.")

    # ── Ollama backend ─────────────────────────────────────────────────────
    backend = OllamaBackend(
        model_name=args.ollama_model,
        host=args.ollama_host,
        timeout=args.timeout,
        max_images_per_request=args.max_images,
        image_quality=args.image_quality,
    )
    description_first_strategy = DescriptionFirstStrategy()

    # ── Eval loop ──────────────────────────────────────────────────────────
    y_true_a, y_pred_a = [], []
    y_true_s, y_pred_s = [], []
    predictions = {}

    weighted_extraction = args.strategy in {
        "cos_two_stage",
        "cos_disambig",
        "description_first",
        "cos_two_stage_description_severity",
        "cos_static_sev",
    }

    for action_id, sample in tqdm(samples.items(), desc=f"[{args.strategy}]"):
        action_hint = ACTION_CLASSES[sample["action"]]

        fpv = extract_all_views_local(
            data_root=args.data_root,
            split=args.split,
            action_id=action_id,
            clip_names=sample["clips"],
            n_frames=args.frames_per_view,
            weighted=weighted_extraction,
            max_views=4,
        )

        if not fpv:
            y_true_a.append(sample["action"])
            y_pred_a.append(-1)
            y_true_s.append(sample["severity"])
            y_pred_s.append(-1)
            predictions[action_id] = {
                "true_action": sample["action"],
                "pred_action": -1,
                "true_severity": sample["severity"],
                "pred_severity": -1,
                "raw_response": "NO FRAMES FOUND",
            }
            continue

        law12_ctx = rag.retrieve(rag.build_query(action_hint))

        try:
            # ── Row 0: static_few_shot ─────────────────────────────────────
            if args.strategy == "static_few_shot":
                prompt = build_static_prompt(len(fpv), law12_ctx)
                raw = backend.classify(fpv, prompt)
                act_idx, sev_idx = parse_response(raw)

            # ── Row 1: data_driven ─────────────────────────────────────────
            elif args.strategy == "data_driven":
                mined_text, mined_imgs = build_examples_text(
                    medoid_cache, n_per_class=1
                )
                prompt = build_data_driven_prompt(
                    len(fpv), law12_ctx, mined_text, severity_priors
                )
                raw = backend.classify(fpv, prompt, extra_images=mined_imgs)
                act_idx, sev_idx = parse_response(raw)

            # ── Row 2: two_stage ───────────────────────────────────────────
            elif args.strategy == "two_stage":
                mined_text, mined_imgs = build_examples_text(
                    medoid_cache, n_per_class=1
                )
                act_prompt = build_two_stage_action_prompt(
                    len(fpv), law12_ctx, mined_text
                )
                raw1 = backend.classify(fpv, act_prompt, extra_images=mined_imgs)
                act_idx = parse_action_only(raw1)
                act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

                sev_text, sev_imgs = build_severity_examples(medoid_cache, act_str)
                sev_law12 = rag.retrieve(rag.build_query(act_str))
                per_action = per_action_priors.get(act_str, severity_priors)
                sev_prompt = build_two_stage_severity_prompt(
                    len(fpv), sev_law12, act_str, sev_text, per_action
                )
                raw2 = backend.classify(fpv, sev_prompt, extra_images=sev_imgs)
                sev_idx = parse_severity_only(raw2)
                raw = f"STAGE1: {raw1}\nSTAGE2: {raw2}"

            # ── Row 4: cos_two_stage ───────────────────────────────────────
            elif args.strategy in {"cos_two_stage", "cos_disambig"}:
                sel_prompt = build_cos_frame_selection_prompt(
                    len(fpv), args.frames_per_view
                )
                raw0 = backend.classify(fpv, sel_prompt)
                key_idx = parse_key_frames(raw0, len(fpv), args.frames_per_view)
                key_fpv = select_key_frames(fpv, key_idx, context_window=1)
                sel_info = format_selected_frame_info(key_idx, len(fpv))

                mined_text, mined_imgs = build_examples_text(
                    medoid_cache, n_per_class=1
                )

                if args.strategy == "cos_disambig":
                    # Add disambiguation text to action prompt
                    from vlm_pipeline.prompts.templates import ACTION_DISAMBIGUATION
                    from vlm_pipeline.prompts.builders import (
                        build_cos_action_disambig_prompt,
                    )

                    act_prompt = build_cos_action_disambig_prompt(
                        len(fpv), law12_ctx, mined_text, sel_info
                    )
                else:
                    act_prompt = build_cos_action_prompt(
                        len(fpv), law12_ctx, mined_text, sel_info
                    )

                raw1 = backend.classify(key_fpv, act_prompt, extra_images=mined_imgs)
                act_idx = parse_action_only(raw1)
                act_str = ACTION_CLASSES[act_idx] if act_idx != -1 else "Dont know"

                sev_text, sev_imgs = build_severity_examples(medoid_cache, act_str)
                sev_law12 = rag.retrieve(rag.build_query(act_str))
                per_action = per_action_priors.get(act_str, severity_priors)
                sev_prompt = build_cos_severity_prompt(
                    len(fpv), sev_law12, act_str, sev_text, per_action, sel_info
                )
                raw2 = backend.classify(key_fpv, sev_prompt, extra_images=sev_imgs)
                sev_idx = parse_severity_only(raw2)
                raw = (
                    f"STAGE0: {raw0}\nSelected: {sel_info}\n"
                    f"STAGE1: {raw1}\nSTAGE2: {raw2}"
                )

            # ── Row X: cos_two_stage_description_severity (NEW) ───────
            elif args.strategy == "cos_two_stage_description_severity":
                act_idx, sev_idx, raw = run_cos_two_stage_description_severity(
                    backend=backend,
                    frames_per_view=fpv,
                    law12_ctx=law12_ctx,
                    medoid_cache=medoid_cache,
                    frames_per_view_count=args.frames_per_view,
                    rag=rag,
                )

            # ── Row X: cos_static_sev (NEW) ──────────────────────────
            elif args.strategy == "cos_static_sev":
                act_idx, sev_idx, raw = run_cos_static_sev(
                    backend=backend,
                    frames_per_view=fpv,
                    law12_ctx=law12_ctx,
                    medoid_cache=medoid_cache,
                    severity_priors=severity_priors,
                    per_action_priors=per_action_priors,
                    frames_per_view_count=args.frames_per_view,
                    rag=rag,
                )

            # ── Row 10: description_first (NEW) ────────────────────────
            elif args.strategy == "description_first":
                act_idx, sev_idx, raw = description_first_strategy.classify(
                    backend=backend,
                    frames_per_view=fpv,
                    law12_ctx=law12_ctx,
                    medoid_cache=medoid_cache,
                    frames_per_view_count=args.frames_per_view,
                )

            else:
                raise ValueError(f"Unknown strategy: {args.strategy}")

        except Exception as e:
            print(f"  Error on {action_id}: {e}")
            act_idx, sev_idx, raw = -1, -1, str(e)

        y_true_a.append(sample["action"])
        y_pred_a.append(act_idx)
        y_true_s.append(sample["severity"])
        y_pred_s.append(sev_idx)
        predictions[action_id] = {
            "true_action": sample["action"],
            "pred_action": act_idx,
            "true_severity": sample["severity"],
            "pred_severity": sev_idx,
            "raw_response": raw,
        }

    # ── Save results ───────────────────────────────────────────────────────
    metrics = compute_metrics(y_true_a, y_pred_a, y_true_s, y_pred_s)
    metrics["strategy"] = args.strategy
    metrics["model_name"] = args.ollama_model
    metrics["predictions"] = predictions

    with open(output_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        k: v
        for k, v in metrics.items()
        if k not in ("predictions", "confusion_action", "confusion_severity")
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS — {args.strategy}")
    print(f"{'='*60}")
    print(f"  Parse rate:    {metrics.get('parse_rate', 0):.1f}%")
    print(f"  Action BA:     {metrics.get('balanced_acc_action', 0):.2f}%")
    print(f"  Severity BA:   {metrics.get('balanced_acc_severity', 0):.2f}%")
    print(f"  Leaderboard:   {metrics.get('leaderboard_value', 0):.4f}")
    print(f"\nSaved to {output_dir}/")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Local VLM ablation via Ollama + raw SoccerNet mp4 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root of SoccerNet dataset (contains Train/, Valid/, Test/)",
    )
    parser.add_argument("--split", default="Valid", choices=["Train", "Valid", "Test"])
    parser.add_argument(
        "--train_annotations",
        default=None,
        help="Path to Train/annotations.json for computing severity priors",
    )
    parser.add_argument(
        "--law12_pdf",
        default=None,
        help="Path to law12.pdf — defaults to <data_root>/law12.pdf",
    )

    # Strategy
    parser.add_argument(
        "--strategy",
        default="static_few_shot",
        choices=[
            "static_few_shot",
            "data_driven",
            "two_stage",
            "cos_two_stage",
            "cos_disambig",
            "description_first",
            "cos_two_stage_description_severity",
            "cos_static_sev",
        ],
        help="Ablation strategy to run",
    )
    parser.add_argument(
        "--medoid_cache",
        default=None,
        help="Path to medoid_cache.json or medoid_cache_v2.json (copy from cluster)",
    )

    # Ollama
    parser.add_argument(
        "--ollama_host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--ollama_model",
        default="qwen2.5vl:7b",
        help="Ollama model tag (run 'ollama list' to see available)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Request timeout in seconds per sample (default: 180)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=16,
        help="Max images per Ollama request (default: 16 = 4 views × 4 frames)",
    )
    parser.add_argument(
        "--image_quality",
        type=int,
        default=85,
        help="JPEG quality for image encoding 1-95 (lower = faster, default: 85)",
    )

    # Eval params
    parser.add_argument(
        "--frames_per_view",
        type=int,
        default=4,
        help="Frames to extract per camera view (default: 4)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)",
    )
    parser.add_argument(
        "--output_dir", default="local_results", help="Directory to save results JSON"
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
