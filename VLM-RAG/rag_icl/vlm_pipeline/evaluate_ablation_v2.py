"""
evaluate_ablation_v2.py
=======================
Thin orchestrator for VLM foul classification ablation experiments.
All logic lives in vlm_pipeline/. This file just handles:
  - CLI argument parsing
  - Loading shared resources (model, RAG, medoid cache, FAISS)
  - The evaluation loop
  - Saving results

Adding a new strategy:
  1. Add implementation to vlm_pipeline/strategies/
  2. Add prompt builders to vlm_pipeline/prompts/builders.py
  3. Add templates to vlm_pipeline/prompts/templates.py
  4. Register strategy name in vlm_pipeline/utils/constants.py ALL_STRATEGIES
  5. Add one elif branch in the eval loop below

STRATEGIES:
  Row 0  static_few_shot      — handcrafted examples, all frames (Qwen baseline)
  Row 1  data_driven          — medoid examples + global severity prior
  Row 2  two_stage            — action first, then severity
  Row 3  rag_icl              — MViT FAISS diverse retrieval
  Row 4  cos_two_stage        — Chain-of-Shot frame selection + two-stage
  Row 5  full_sev_two_stage   — CoS action + ALL frames for severity ← NEW
  Row 6  per_action_prior     — static_few_shot + per-action severity calibration ← NEW
  Row 7  targeted_retrieval   — two-stage with targeted medoid examples ← NEW
  Row 8  ordinal_severity     — two-stage with ordinal force reasoning ← NEW
  Row 9  cos_full_sev         — best combined: CoS + full frames + ordinal ← NEW

Usage:
  python evaluate_ablation_v2.py --strategy cos_full_sev --max_samples 50
  python evaluate_ablation_v2.py --strategy full_sev_two_stage
  sbatch run_ablation_v2.sh full_sev_two_stage 50
  sbatch run_ablation_v2.sh cos_full_sev
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import h5py

# Add parent dir to path so vlm_pipeline is importable
sys.path.insert(0, str(Path(__file__).parent))

from vlm_pipeline.utils import (
    load_annotations,
    compute_metrics,
    compute_severity_priors,
    compute_per_action_severity_priors,
    extract_all_views,
    parse_key_frames,
    select_key_frames,
    format_selected_frame_info,
    ACTION_CLASSES,
    SEVERITY_CLASSES,
    PER_ACTION_SEVERITY_PRIOR,
    SEVERITY_PRIOR_DEFAULT,
    ALL_STRATEGIES,
)
from vlm_pipeline.retrieval import (
    Law12RAG,
    load_medoid_cache,
    build_medoid_cache,
    build_examples_text,
    build_targeted_examples,
    build_severity_examples,
    MViTRetriever,
)
from vlm_pipeline.prompts import (
    build_static_prompt,
    build_data_driven_prompt,
    build_two_stage_action_prompt,
    build_two_stage_severity_prompt,
    build_ragicl_prompt,
    build_cos_frame_selection_prompt,
    build_cos_action_disambig_prompt,
    build_cos_severity_prompt,
    build_full_frame_severity_prompt,
    build_per_action_prior_prompt,
    build_targeted_retrieval_prompt,
    build_ordinal_severity_prompt,
)
from vlm_pipeline.strategies.base import (
    parse_response,
    parse_action_only,
    parse_severity_only,
)
from vlm_pipeline.strategies.severity_focused import (
    run_full_sev_two_stage,
    run_per_action_prior,
    run_targeted_retrieval,
    run_ordinal_severity,
    run_cos_full_sev,
)
from vlm_pipeline.backends import get_backend

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Strategy: {args.strategy}")
    print(f"Model:    {args.model_name}")
    print(f"{'='*60}")

    # ── Load annotations ──────────────────────────────────────────────────────
    samples = load_annotations(args.annotations)
    if args.max_samples:
        ids = list(samples.keys())[: args.max_samples]
        samples = {k: samples[k] for k in ids}
    print(f"Evaluating on {len(samples)} samples.")

    # ── Law 12 RAG ────────────────────────────────────────────────────────────
    rag = Law12RAG(pdf_path=args.law12_pdf, top_k=3, use_embeddings=True)

    # ── Severity priors ───────────────────────────────────────────────────────
    if args.train_annotations and Path(args.train_annotations).exists():
        severity_priors = compute_severity_priors(args.train_annotations)
        per_action_priors = compute_per_action_severity_priors(args.train_annotations)
    else:
        severity_priors = SEVERITY_PRIOR_DEFAULT
        per_action_priors = PER_ACTION_SEVERITY_PRIOR
        print("[Priors] Using hardcoded defaults")

    # ── Medoid cache ──────────────────────────────────────────────────────────
    medoid_cache = None
    needs_medoid = args.strategy in {
        "data_driven",
        "two_stage",
        "cos_two_stage",
        "full_sev_two_stage",
        "targeted_retrieval",
        "ordinal_severity",
        "cos_full_sev",
    }
    if needs_medoid:
        if not args.medoid_cache or not Path(args.medoid_cache).exists():
            raise FileNotFoundError(
                f"Medoid cache not found: {args.medoid_cache}\n"
                "Run with --build_medoid_cache first."
            )
        medoid_cache = load_medoid_cache(args.medoid_cache)
        print(f"[MedoidCache] Loaded {len(medoid_cache)} entries.")

    # ── FAISS retriever ───────────────────────────────────────────────────────
    retriever = None
    if args.strategy == "rag_icl":
        retriever = MViTRetriever(args.faiss_index_path, args.faiss_meta_path)

    # ── VLM backend ───────────────────────────────────────────────────────────
    backend = get_backend(args.model_name)

    # ── Eval loop ─────────────────────────────────────────────────────────────
    y_true_a, y_pred_a = [], []
    y_true_s, y_pred_s = [], []
    predictions = {}

    with h5py.File(args.hdf5_path, "r", swmr=True) as hdf5:
        for action_id, sample in tqdm(samples.items(), desc=f"  [{args.strategy}]"):
            action_key = f"action_{action_id}"
            action_hint = ACTION_CLASSES[sample["action"]]

            # Load frames (weighted sampling for strategies that benefit)
            weighted = args.strategy in {
                "cos_two_stage",
                "full_sev_two_stage",
                "cos_full_sev",
            }
            fpv = extract_all_views(
                hdf5,
                action_key,
                sample["clips"],
                n_frames=args.frames_per_view,
                weighted=weighted,
                max_views=4,
            )

            if not fpv:
                y_true_a.append(sample["action"])
                y_pred_a.append(-1)
                y_true_s.append(sample["severity"])
                y_pred_s.append(-1)
                continue

            law12_ctx = rag.retrieve(rag.build_query(action_hint))

            try:
                # ── Row 0: static_few_shot ─────────────────────────────────
                if args.strategy == "static_few_shot":
                    prompt = build_static_prompt(len(fpv), law12_ctx)
                    raw = backend.classify(fpv, prompt)
                    act_idx, sev_idx = parse_response(raw)

                # ── Row 1: data_driven ─────────────────────────────────────
                elif args.strategy == "data_driven":
                    mined_text, mined_imgs = build_examples_text(
                        medoid_cache, n_per_class=1
                    )
                    prompt = build_data_driven_prompt(
                        len(fpv), law12_ctx, mined_text, severity_priors
                    )
                    raw = backend.classify(fpv, prompt, extra_images=mined_imgs)
                    act_idx, sev_idx = parse_response(raw)

                # ── Row 2: two_stage ───────────────────────────────────────
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

                # ── Row 3: rag_icl ─────────────────────────────────────────
                elif args.strategy == "rag_icl":
                    dynamic_ex = retriever.retrieve(fpv[0], k=args.retrieval_k)
                    prompt = build_ragicl_prompt(len(fpv), law12_ctx, dynamic_ex)
                    raw = backend.classify(fpv, prompt)
                    act_idx, sev_idx = parse_response(raw)

                # ── Row 4: cos_two_stage ───────────────────────────────────
                elif args.strategy == "cos_two_stage":
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
                    act_prompt = build_cos_action_disambig_prompt(
                        len(fpv), law12_ctx, mined_text, sel_info
                    )
                    raw1 = backend.classify(
                        key_fpv, act_prompt, extra_images=mined_imgs
                    )
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

                # ── Row 5: full_sev_two_stage (NEW) ───────────────────────
                elif args.strategy == "full_sev_two_stage":
                    act_idx, sev_idx, raw = run_full_sev_two_stage(
                        backend=backend,
                        frames_per_view=fpv,
                        law12_ctx=law12_ctx,
                        medoid_cache=medoid_cache,
                        severity_priors=severity_priors,
                        frames_per_view_count=args.frames_per_view,
                        action_hint=action_hint,
                        rag=rag,
                    )

                # ── Row 6: per_action_prior (NEW) ─────────────────────────
                elif args.strategy == "per_action_prior":
                    act_idx, sev_idx, raw = run_per_action_prior(
                        backend=backend,
                        frames_per_view=fpv,
                        law12_ctx=law12_ctx,
                        per_action_priors=per_action_priors,
                    )

                # ── Row 7: targeted_retrieval (NEW) ───────────────────────
                elif args.strategy == "targeted_retrieval":
                    act_idx, sev_idx, raw = run_targeted_retrieval(
                        backend=backend,
                        frames_per_view=fpv,
                        law12_ctx=law12_ctx,
                        medoid_cache=medoid_cache,
                        severity_priors=severity_priors,
                        rag=rag,
                    )

                # ── Row 8: ordinal_severity (NEW) ─────────────────────────
                elif args.strategy == "ordinal_severity":
                    act_idx, sev_idx, raw = run_ordinal_severity(
                        backend=backend,
                        frames_per_view=fpv,
                        law12_ctx=law12_ctx,
                        medoid_cache=medoid_cache,
                        severity_priors=severity_priors,
                        rag=rag,
                    )

                # ── Row 9: cos_full_sev (NEW — best combined) ─────────────
                elif args.strategy == "cos_full_sev":
                    act_idx, sev_idx, raw = run_cos_full_sev(
                        backend=backend,
                        frames_per_view=fpv,
                        law12_ctx=law12_ctx,
                        medoid_cache=medoid_cache,
                        severity_priors=severity_priors,
                        frames_per_view_count=args.frames_per_view,
                        rag=rag,
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

    # ── Save results ──────────────────────────────────────────────────────────
    metrics = compute_metrics(y_true_a, y_pred_a, y_true_s, y_pred_s)
    metrics["strategy"] = args.strategy
    metrics["model_name"] = args.model_name
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
        description="VLM foul classification ablation v2 (modular pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--build_medoid_cache",
        action="store_true",
        help="Build medoid cache from training data and exit.",
    )

    parser.add_argument("--strategy", choices=ALL_STRATEGIES, default="static_few_shot")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")

    # Data paths
    parser.add_argument("--hdf5_path")
    parser.add_argument("--annotations")
    parser.add_argument("--train_hdf5")
    parser.add_argument("--train_annotations")
    parser.add_argument("--law12_pdf", default=None)

    # FAISS / medoid
    parser.add_argument("--faiss_index_path", default=None)
    parser.add_argument("--faiss_meta_path", default=None)
    parser.add_argument(
        "--medoid_cache",
        default="/net/tscratch/people/plgaszos/vlm_rag_icl/medoid_cache.json",
    )

    # Eval params
    parser.add_argument("--frames_per_view", type=int, default=4)
    parser.add_argument("--retrieval_k", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="ablation_results_v2/output")

    args = parser.parse_args()

    if args.build_medoid_cache:
        if not all(
            [
                args.train_hdf5,
                args.train_annotations,
                args.faiss_index_path,
                args.faiss_meta_path,
            ]
        ):
            parser.error(
                "--build_medoid_cache requires: "
                "--train_hdf5, --train_annotations, "
                "--faiss_index_path, --faiss_meta_path"
            )
        build_medoid_cache(
            train_hdf5_path=args.train_hdf5,
            train_annotations_path=args.train_annotations,
            faiss_index_path=args.faiss_index_path,
            faiss_meta_path=args.faiss_meta_path,
            output_path=args.medoid_cache,
        )
        return

    if not args.hdf5_path or not args.annotations:
        parser.error("Evaluation requires --hdf5_path and --annotations")

    evaluate(args)


if __name__ == "__main__":
    main()
