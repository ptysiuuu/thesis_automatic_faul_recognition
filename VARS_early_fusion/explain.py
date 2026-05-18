"""
VARS Explainability Pipeline: Three-stage system for interpretable foul recognition.

Stage 1: Evidence Extraction     → capture all model signals (temporal attention, logits, auxiliary outputs)
Stage 2: Evidence Interpretation → convert raw signals to structured text (confidence, temporal focus, physical indicators)
Stage 3: LLM Explanation         → generate natural language explanation (Gemini API or local LLM)

Usage:
    python explain.py \
        --path /path/to/dataset \
        --checkpoint /path/to/model.pt \
        --split test \
        --num_views 5 \
        --batch_size 1 \
        --output_dir ./explanations \
        --llm_backend gemini \
        --gemini_api_key <YOUR_KEY>
"""

import os
import json
import logging
import torch
import torch.nn.functional as F
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm

from dataset import MultiViewDataset
from model import MVNetwork, EarlyFusionNetwork
from train import ordinal_predict, ordinal_to_probs, EMA
from config.classes import EVENT_DICTIONARY, INVERSE_EVENT_DICTIONARY
from torchvision.models.video import MViT_V2_S_Weights

# ============================================================================
# STAGE 1: Evidence Extraction
# ============================================================================


class EvidenceExtractor:
    """
    Captures all intermediate signals from VARS inference.

    Signals captured per sample:
    - Action: logits (8-dim), probabilities, top-2 predictions
    - Severity: ordinal logits (3-dim), sigmoid probabilities, class prediction
    - Auxiliary: contact, bodypart, try_to_play, handball (logits + probs)
    - Temporal attention: weights per view [B, V, T'] with frame mapping
    - View quality: reliability scores per view [B, V]
    - Metadata: action_id, game_id, etc.
    """

    def __init__(
        self,
        num_views: int = 5,
        T_per_view: int = 16,
        fps: int = 12,
        clip_start_frame: int = 58,
        clip_end_frame: int = 92,
    ):
        self.num_views = num_views
        self.T_per_view = T_per_view
        self.fps = fps
        self.num_temporal_tokens = 8  # standard for most backbones
        self.clip_start_frame = clip_start_frame
        self.clip_end_frame = clip_end_frame

    def extract(
        self,
        batch: Tuple,
        model_output: Tuple,
        model: torch.nn.Module,
    ) -> Dict[str, Any]:
        """
        Extract evidence from a single batch item (batch_size=1 assumed).

        Args:
            batch: Dataset batch tuple (includes mvclips, action_ids, etc.)
            model_output: 7-tuple from model.forward():
                         (severity_logits, action_logits, contact, bodypart,
                          try_to_play, handball, attention_weights)
            model: The model instance (to check if early_fusion or mv)

        Returns:
            Dict containing all evidence signals
        """
        # Unpack model output
        (
            severity_logits,
            action_logits,
            contact_logit,
            bodypart_logit,
            try_to_play_logit,
            handball_logit,
            attention_weights,
        ) = model_output

        # Action: full probability distribution
        action_probs = F.softmax(action_logits, dim=1)  # [B, 8]
        action_pred = torch.argmax(action_logits, dim=1)  # [B]
        action_top2_probs, action_top2_ids = torch.topk(action_probs, k=2, dim=1)
        action_confidence_gap = (
            action_top2_probs[:, 0] - action_top2_probs[:, 1]
        ).item()

        # Severity: ordinal interpretation
        severity_probs = torch.sigmoid(
            severity_logits
        )  # [B, 3] cumulative probabilities
        severity_pred = ordinal_predict(severity_logits)  # [B] class 0-3
        severity_probs_full = ordinal_to_probs(severity_logits)  # [B, 4] per-class

        # Auxiliary: binary predictions with probabilities
        contact_prob = torch.sigmoid(contact_logit.unsqueeze(-1))  # [B, 1]
        bodypart_prob = torch.sigmoid(bodypart_logit.unsqueeze(-1))
        try_to_play_prob = torch.sigmoid(try_to_play_logit.unsqueeze(-1))
        handball_prob = torch.sigmoid(handball_logit.unsqueeze(-1))

        # Temporal attention weights (if available)
        temporal_evidence = None
        if attention_weights is not None:
            print(f"DEBUG attention_weights shape: {attention_weights.shape}")
            print(f"DEBUG attention_weights values: {attention_weights}")
        if attention_weights is not None:
            temporal_evidence = self._process_temporal_attention(
                attention_weights, is_early_fusion=isinstance(model, EarlyFusionNetwork)
            )

        # Extract metadata from batch
        action_ids = batch[7]  # action_ids from dataset loader
        action_id = (
            action_ids[0].item() if torch.is_tensor(action_ids[0]) else action_ids[0]
        )

        evidence = {
            # Predictions
            "action": {
                "prediction": action_pred[0].item(),
                "prediction_name": INVERSE_EVENT_DICTIONARY.get("action_class", {}).get(
                    action_pred[0].item(), "UNKNOWN"
                ),
                "logits": action_logits[0].cpu().detach().tolist(),
                "probabilities": action_probs[0].cpu().detach().tolist(),
                "confidence": action_probs[0, action_pred[0]].item(),
                "top2": {
                    "ids": action_top2_ids[0].cpu().tolist(),
                    "names": [
                        INVERSE_EVENT_DICTIONARY.get("action_class", {}).get(
                            int(id_), "UNKNOWN"
                        )
                        for id_ in action_top2_ids[0]
                    ],
                    "probs": action_top2_probs[0].cpu().detach().tolist(),
                    "confidence_gap": action_confidence_gap,
                },
            },
            "severity": {
                "prediction": severity_pred[0].item(),
                "ordinal_logits": severity_logits[0].cpu().detach().tolist(),
                "ordinal_probs": severity_probs[0]
                .cpu()
                .detach()
                .tolist(),  # cumulative
                "class_probs": severity_probs_full[0]
                .cpu()
                .detach()
                .tolist(),  # per-class
                "confidence": severity_probs_full[0, severity_pred[0]].item(),
            },
            "auxiliary": {
                "contact": {
                    "logit": contact_logit[0].item(),
                    "probability": contact_prob[0, 0].item(),
                    "prediction": (contact_prob[0, 0] > 0.5).item(),
                },
                "bodypart": {
                    "logit": bodypart_logit[0].item(),
                    "probability": bodypart_prob[0, 0].item(),
                    "prediction": (bodypart_prob[0, 0] > 0.5).item(),
                },
                "try_to_play": {
                    "logit": try_to_play_logit[0].item(),
                    "probability": try_to_play_prob[0, 0].item(),
                    "prediction": (try_to_play_prob[0, 0] > 0.5).item(),
                },
                "handball": {
                    "logit": handball_logit[0].item(),
                    "probability": handball_prob[0, 0].item(),
                    "prediction": (handball_prob[0, 0] > 0.5).item(),
                },
            },
            "temporal": temporal_evidence,
            "metadata": {
                "action_id": action_id,
                "num_views": self.num_views,
                "frames_per_view": self.T_per_view,
            },
        }

        return evidence

    def _process_temporal_attention(
        self, attention_weights, is_early_fusion: bool = False
    ) -> Dict:
        """
        Convert temporal attention weights to interpretable signals.

        For MVNetwork: attention_weights = [B, V, T'] temporal weights from TransformerAggregate
        For EarlyFusionNetwork: attention_weights = None (no view dimension)

        Maps each temporal token to corresponding frame number and computes:
        - Center of mass (which moment the model focused on)
        - Entropy (confidence of temporal localization)
        - Per-view distributions
        """
        if attention_weights is None:
            return None

        weights = attention_weights.detach()

        # Accept only temporal attention-like tensors [B, V, T].
        # Some aggregators may return non-temporal attention shapes.
        if weights.dim() == 2:
            # [B, T] -> treat as single-view temporal attention
            weights = weights.unsqueeze(1)
        elif weights.dim() != 3:
            return {
                "has_signal": False,
                "reason": f"Unsupported attention tensor shape: {tuple(weights.shape)}",
            }

        B, V, T = weights.shape

        # If attention is transposed as [B, T, V], fix to [B, V, T].
        if V == self.num_temporal_tokens and T == self.num_views:
            weights = weights.transpose(1, 2)
            B, V, T = weights.shape

        # Non-temporal attention (e.g., one scalar per view) is not useful for temporal analysis.
        if T <= 1:
            return {
                "has_signal": False,
                "reason": f"Attention has only {T} temporal token(s); temporal localization requires >1.",
            }

        # Map token 0..T-1 to the configured frame range [clip_start_frame, clip_end_frame].
        if T == 1:
            token_frames = [self.clip_start_frame]
        else:
            token_frames = [
                int(
                    round(
                        self.clip_start_frame
                        + (i / (T - 1)) * (self.clip_end_frame - self.clip_start_frame)
                    )
                )
                for i in range(T)
            ]

        # Multi-view temporal case: [B, V, T]
        if weights.dim() == 3:
            weights = weights[0]  # [V, T]

            # Compute per-view statistics
            view_stats = []
            for v_idx in range(V):
                w = weights[v_idx]  # [T]
                # Ensure a proper distribution even if upstream tensor is not perfectly normalized.
                w = w / (w.sum() + 1e-8)
                com = (w * torch.arange(T, device=w.device).float()).sum() / w.sum()
                entropy = -(w * (w + 1e-8).log()).sum().item()
                peak_token = torch.argmax(w).item()
                peak_frame = token_frames[peak_token]

                view_stats.append(
                    {
                        "view_id": v_idx,
                        "weights": w.cpu().detach().tolist(),
                        "peak_token": peak_token,
                        "peak_frame": peak_frame,
                        "center_of_mass": com.item(),
                        "entropy": entropy,
                    }
                )

            # Aggregate across views
            avg_weights = weights.mean(dim=0)
            avg_weights = avg_weights / (avg_weights.sum() + 1e-8)
            com_agg = (
                avg_weights * torch.arange(T, device=avg_weights.device).float()
            ).sum() / avg_weights.sum()
            entropy_agg = -(avg_weights * (avg_weights + 1e-8).log()).sum().item()
            peak_token_agg = torch.argmax(avg_weights).item()
            peak_frame_agg = token_frames[peak_token_agg]

            return {
                "has_signal": True,
                "num_views": V,
                "token_frame_mapping": dict(enumerate(token_frames)),
                "per_view": view_stats,
                "aggregated": {
                    "weights": avg_weights.cpu().detach().tolist(),
                    "peak_token": peak_token_agg,
                    "peak_frame": peak_frame_agg,
                    "center_of_mass": com_agg.item(),
                    "entropy": entropy_agg,
                },
            }
        return {
            "has_signal": False,
            "reason": "No temporal attention weights",
        }


# ============================================================================
# Inference function (enhanced from train.py)
# ============================================================================


def inference_with_signals(
    dataloader,
    model,
    ema: Optional[EMA] = None,
    use_tta: bool = True,
    evidence_extractor: Optional[EvidenceExtractor] = None,
) -> List[Dict[str, Any]]:
    """
    Run inference and collect all evidence signals.

    Returns:
        List of evidence dictionaries, one per sample
    """
    if ema is not None:
        ema.apply_shadow()

    model.eval()
    all_evidence = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference + Signal Extraction"):
            mvclips = batch[6].cuda().float()

            # Forward pass
            model_output = model(mvclips)

            # Extract evidence
            if evidence_extractor is not None:
                evidence = evidence_extractor.extract(batch, model_output, model)
                all_evidence.append(evidence)

    if ema is not None:
        ema.restore()

    return all_evidence


# ============================================================================
# Main execution
# ============================================================================


def main(args):
    """
    Execute the full three-stage explainability pipeline.

    Stage 1: Evidence extraction (this script)
    Stage 2: Evidence interpretation (evidence_interpreter.py)
    Stage 3: LLM explanation generation (llm_interface.py)
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ===== Load model =====
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Infer model type from checkpoint or args
    is_fusion_mode = args.fusion_mode or (
        checkpoint.get("config", {}).get("fusion_mode", False)
    )

    if is_fusion_mode:
        model = EarlyFusionNetwork(
            num_views=args.num_views,
            T_per_view=args.frames_per_view,
        ).cuda()
    else:
        model = MVNetwork(
            net_name=args.backbone,
            agr_type=args.aggregation,
            graph_topology=args.graph_topology,
        ).cuda()

    model.load_state_dict(checkpoint["state_dict"], strict=False)
    logger.info(f"Model loaded (fusion_mode={is_fusion_mode})")

    # Load EMA if available
    ema = None
    if args.use_ema and "ema" in checkpoint:
        ema = EMA(model, decay=0.999)
        ema.load_state_dict(checkpoint["ema"])
        logger.info("EMA state loaded")

    # ===== Load dataset =====
    logger.info(f"Loading {args.split} split from {args.path}")
    from main import TAdaFormerTransform

    dataset = MultiViewDataset(
        path=args.path,
        split=args.split.capitalize(),
        num_views=args.num_views,
        start=args.start_frame,
        end=args.end_frame,
        fps=args.fps,
        transform_model=TAdaFormerTransform(size=224),
        fusion_mode=is_fusion_mode,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.max_num_worker,
        pin_memory=True,
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Optionally limit the number of examples processed (useful for quick tests)
    if getattr(args, "num_examples", None) is not None and args.num_examples > 0:
        limit = min(len(dataset), args.num_examples)
        logger.info(
            f"Limiting dataset to first {limit} samples (requested {args.num_examples})"
        )
        dataset = torch.utils.data.Subset(dataset, list(range(limit)))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.max_num_worker,
            pin_memory=True,
        )
        logger.info(f"Dataset limited: {len(dataset)} samples")

    # ===== STAGE 1: Evidence Extraction =====
    logger.info("=" * 70)
    logger.info("STAGE 1: Evidence Extraction")
    logger.info("=" * 70)

    extractor = EvidenceExtractor(
        num_views=args.num_views,
        T_per_view=args.frames_per_view,
        fps=args.fps,
        clip_start_frame=args.start_frame,
        clip_end_frame=args.end_frame,
    )

    all_evidence = inference_with_signals(
        dataloader,
        model,
        ema=ema,
        use_tta=args.use_tta,
        evidence_extractor=extractor,
    )

    logger.info(f"Extracted evidence from {len(all_evidence)} samples")

    # Save raw evidence
    evidence_file = output_dir / f"evidence_{args.split}.json"
    with open(evidence_file, "w") as f:
        json.dump(all_evidence, f, indent=2)
    logger.info(f"Evidence saved to {evidence_file}")

    # ===== STAGE 2: Evidence Interpretation =====
    logger.info("=" * 70)
    logger.info("STAGE 2: Evidence Interpretation")
    logger.info("=" * 70)

    from evidence_interpreter import EvidenceInterpreter

    interpreter = EvidenceInterpreter()
    interpreted_evidence = []

    for i, evidence in enumerate(tqdm(all_evidence, desc="Interpreting evidence")):
        interpreted = interpreter.interpret(evidence)
        interpreted_evidence.append(interpreted)

    # Save interpreted evidence
    interpreted_file = output_dir / f"interpreted_{args.split}.json"
    with open(interpreted_file, "w") as f:
        json.dump(interpreted_evidence, f, indent=2)
    logger.info(f"Interpreted evidence saved to {interpreted_file}")

    # ===== STAGE 3: LLM Explanation Generation =====
    if not args.skip_llm:
        logger.info("=" * 70)
        logger.info("STAGE 3: LLM Explanation Generation")
        logger.info("=" * 70)

        from llm_interface import get_llm_interface

        use_video = getattr(args, "use_video", True) and args.llm_backend == "gemini"
        llm = get_llm_interface(
            backend=args.llm_backend,
            model=args.llm_model,
            api_key=args.gemini_api_key,
            temperature=args.llm_temperature,
            use_video=use_video,
        )

        logger.info(f"Using LLM backend: {args.llm_backend} (video={use_video})")

        explanations = []
        for i, (evidence, interpreted) in enumerate(
            tqdm(
                zip(all_evidence, interpreted_evidence),
                total=len(all_evidence),
                desc="Generating explanations",
            )
        ):
            video_paths = None
            if use_video:
                action_id = evidence.get("metadata", {}).get("action_id")
                if action_id is not None:
                    clip_dir = os.path.join(
                        args.path, args.split.capitalize(), f"action_{action_id}"
                    )
                    video_paths = [
                        os.path.join(clip_dir, f"clip_{v}.mp4")
                        for v in range(args.num_views)
                        if os.path.exists(os.path.join(clip_dir, f"clip_{v}.mp4"))
                    ]

            explanation = llm.generate_explanation(evidence, interpreted, video_paths=video_paths)
            explanations.append(explanation)

        # Save explanations
        explanations_file = output_dir / f"explanations_{args.split}.json"
        with open(explanations_file, "w") as f:
            json.dump(explanations, f, indent=2)
        logger.info(f"Explanations saved to {explanations_file}")

    # ===== Summary =====
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Raw evidence: {evidence_file}")
    logger.info(f"Interpreted evidence: {interpreted_file}")
    if not args.skip_llm:
        logger.info(f"LLM explanations: {explanations_file}")
    logger.info(f"Total samples processed: {len(all_evidence)}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="VARS Explainability Pipeline: Evidence Extraction → Interpretation → LLM Generation",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--path", required=True, type=str, help="Path to dataset root")
    parser.add_argument(
        "--split", default="test", type=str, choices=["test", "chall", "valid", "train"]
    )
    parser.add_argument("--start_frame", default=58, type=int)
    parser.add_argument("--end_frame", default=92, type=int)
    parser.add_argument("--fps", default=12, type=int)
    parser.add_argument("--num_views", default=5, type=int)
    parser.add_argument("--frames_per_view", default=16, type=int)

    # Model
    parser.add_argument(
        "--checkpoint", required=True, type=str, help="Path to trained checkpoint"
    )
    parser.add_argument("--fusion_mode", action="store_true", default=False)
    parser.add_argument("--backbone", default="mvit_v2_s", type=str)
    parser.add_argument("--aggregation", default="transformer", type=str)
    parser.add_argument("--graph_topology", default="structured", type=str)
    parser.add_argument("--use_ema", action="store_true", default=False)
    parser.add_argument("--use_tta", action="store_true", default=False)

    # Inference
    parser.add_argument("--max_num_worker", default=4, type=int)

    # LLM
    parser.add_argument(
        "--skip_llm",
        action="store_true",
        default=False,
        help="Skip Stage 3 LLM generation",
    )
    parser.add_argument(
        "--llm_backend",
        default="gemini",
        type=str,
        choices=["gemini", "local", "anthropic"],
    )
    parser.add_argument(
        "--llm_model",
        default="",
        type=str,
        help="Specific model (e.g., 'mistral-7b-instruct')",
    )
    parser.add_argument(
        "--gemini_api_key",
        default="",
        type=str,
        help="Gemini API key (or use GEMINI_API_KEY env var)",
    )
    parser.add_argument("--llm_temperature", default=0.7, type=float)
    parser.add_argument(
        "--use_video",
        action="store_true",
        default=True,
        help="Send video clips to Gemini (native video input, gemini backend only)",
    )
    parser.add_argument(
        "--no_video",
        dest="use_video",
        action="store_false",
        help="Disable video input; send text evidence only",
    )

    # Output
    parser.add_argument("--output_dir", default="./explanations", type=str)
    parser.add_argument(
        "--num_examples",
        default=None,
        type=int,
        help="Maximum number of examples to process (None = all)",
    )

    args = parser.parse_args()
    main(args)
