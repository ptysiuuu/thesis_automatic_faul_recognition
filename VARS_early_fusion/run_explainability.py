#!/usr/bin/env python3
"""
VARS Explainability Quick-Start Launcher

Simple script to run the full explainability pipeline with sensible defaults.

Usage:
    python run_explainability.py --checkpoint path/to/model.pth --dataset_path /path/to/dataset
    python run_explainability.py --help
"""

import argparse
import os
import json
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run VARS Explainability Pipeline (Evidence → Interpretation → Explanation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Essential arguments
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to trained VARS model checkpoint",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
        help="Path to SoccerNet dataset root",
    )

    # Model config
    parser.add_argument("--split", default="test", choices=["test", "chall", "valid"])
    parser.add_argument("--num_views", default=5, type=int)
    parser.add_argument("--frames_per_view", default=16, type=int)
    parser.add_argument("--fps", default=12, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_num_worker", default=4, type=int)

    # Model architecture (auto-detect from checkpoint if possible)
    parser.add_argument("--fusion_mode", action="store_true", help="Early fusion mode")
    parser.add_argument("--backbone", default="mvit_v2_s", help="For multi-view mode")
    parser.add_argument(
        "--aggregation", default="transformer", help="For multi-view mode"
    )

    # LLM config
    parser.add_argument(
        "--llm_backend",
        default="gemini",
        choices=["gemini", "anthropic", "local", "none"],
        help="'none' skips explanation generation (Stage 3)",
    )
    parser.add_argument(
        "--llm_model",
        default="",
        help="Specific LLM model (required if --llm_backend=local)",
    )
    parser.add_argument(
        "--gemini_api_key",
        default="",
        help="Gemini API key (or set GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--anthropic_api_key",
        default="",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument("--llm_temperature", default=0.7, type=float)

    # Output
    parser.add_argument("--output_dir", default="./explanations", type=str)

    # Evaluation
    parser.add_argument(
        "--create_eval_template",
        action="store_true",
        help="Create manual evaluation template for human review",
    )
    parser.add_argument("--eval_sample_size", default=30, type=int)

    # Other
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--num_examples",
        default=None,
        type=int,
        help="Maximum number of examples to process (None = all)",
    )

    args = parser.parse_args()

    # ===== Validate inputs =====
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return 1

    if not Path(args.dataset_path).exists():
        logger.error(f"Dataset path not found: {args.dataset_path}")
        return 1

    if args.llm_backend == "local" and not args.llm_model:
        logger.warning(
            "Local LLM requires --llm_model. Defaulting to Mistral-7B-Instruct"
        )
        args.llm_model = "mistralai/Mistral-7B-Instruct-v0.2"

    # ===== Check API keys if needed =====
    if args.llm_backend == "gemini":
        key = args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            logger.error(
                "Gemini API key required. Provide via --gemini_api_key or GOOGLE_API_KEY env var"
            )
            return 1

    # ===== Run pipeline =====
    logger.info("=" * 70)
    logger.info("VARS EXPLAINABILITY PIPELINE")
    logger.info("=" * 70)

    # Import main function
    from explain import main as run_explain

    # Construct args for explain.py
    explain_args = argparse.Namespace(
        path=args.dataset_path,
        checkpoint=args.checkpoint,
        num_examples=args.num_examples,
        split=args.split,
        start_frame=0,
        end_frame=125,
        fps=args.fps,
        num_views=args.num_views,
        frames_per_view=args.frames_per_view,
        fusion_mode=args.fusion_mode,
        pre_model=args.backbone,
        backbone=args.backbone,
        aggregation=args.aggregation,
        graph_topology="structured",
        use_ema=args.use_ema,
        use_tta=args.use_tta,
        max_num_worker=args.max_num_worker,
        skip_llm=(args.llm_backend == "none"),
        llm_backend=args.llm_backend if args.llm_backend != "none" else "gemini",
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        gemini_api_key=args.gemini_api_key,
        output_dir=args.output_dir,
    )

    try:
        run_explain(explain_args)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    # ===== Post-processing: Create evaluation template if requested =====
    if args.create_eval_template:
        logger.info("=" * 70)
        logger.info("Creating Manual Evaluation Template")
        logger.info("=" * 70)

        from evaluation_framework import QualitativeEvaluator

        output_dir = Path(args.output_dir)

        # Load all results
        evidence_file = output_dir / f"evidence_{args.split}.json"
        interpreted_file = output_dir / f"interpreted_{args.split}.json"
        explanations_file = output_dir / f"explanations_{args.split}.json"

        if not all(
            [
                evidence_file.exists(),
                interpreted_file.exists(),
                explanations_file.exists(),
            ]
        ):
            logger.error("Missing output files for evaluation template")
            return 1

        with open(evidence_file) as f:
            evidence = json.load(f)
        with open(interpreted_file) as f:
            interpreted = json.load(f)
        with open(explanations_file) as f:
            explanations = json.load(f)

        eval_template_file = output_dir / f"manual_evaluation_{args.split}.json"

        QualitativeEvaluator.create_evaluation_json(
            evidence,
            interpreted,
            explanations,
            eval_template_file,
            sample_size=args.eval_sample_size,
        )

        logger.info(f"Evaluation template created: {eval_template_file}")
        logger.info("Please score each sample on 1-5 scale for:")
        logger.info("  - Factuality: Does explanation match evidence?")
        logger.info("  - Completeness: Does it use all key signals?")
        logger.info("  - Clarity: Is it understandable?")
        logger.info("  - Specificity: Does it cite specific numbers/signals?")
        logger.info("  - Law 12 Correctness: Is legal reasoning sound?")

    # ===== Summary =====
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Files generated:")
    logger.info(f"  - evidence_{args.split}.json (raw signals)")
    logger.info(f"  - interpreted_{args.split}.json (structured interpretation)")
    if args.llm_backend != "none":
        logger.info(f"  - explanations_{args.split}.json (LLM-generated)")
    if args.create_eval_template:
        logger.info(f"  - manual_evaluation_{args.split}.json (for human review)")

    logger.info("\nNext steps:")
    logger.info("1. Review explanations_*.json for quality")
    if args.create_eval_template:
        logger.info(
            f"2. Have a domain expert score manual_evaluation_{args.split}.json"
        )
        logger.info("3. Compute evaluation summary: ")
        logger.info(
            f'   python -c "from evaluation_framework import QualitativeEvaluator; '
        )
        logger.info(f"   from pathlib import Path; ")
        logger.info(
            f"   summary = QualitativeEvaluator.summarize_evaluations(Path('{args.output_dir}/manual_evaluation_{args.split}.json')); "
        )
        logger.info(f'   import json; print(json.dumps(summary, indent=2))"')

    return 0


if __name__ == "__main__":
    exit(main())
