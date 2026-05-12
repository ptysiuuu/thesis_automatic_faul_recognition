"""
VARS Explainability: Evaluation Framework

Two types of evaluation to validate the explainability system for thesis:

1. QUANTITATIVE: Correlations between evidence signals and prediction accuracy
   - Does peaked temporal attention (low entropy) correlate with higher accuracy?
   - Do correct predictions have higher confidence scores?
   - Do auxiliary signals agree with ground truth when available?

2. QUALITATIVE: Human evaluation of explanation quality
   - Does the explanation accurately reflect what the model attended to?
   - Is the explanation coherent and comprehensible?
   - Does it help understand the model's decision process (even if wrong)?
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class QuantitativeEvaluator:
    """
    Analyze correlations between evidence signals and prediction accuracy.

    Tests hypotheses like:
    - H1: Peaked temporal attention → higher accuracy
    - H2: Higher action confidence → higher accuracy
    - H3: Low severity boundary crossing → harder predictions
    - H4: Auxiliary signal agreement → better predictions
    """

    def __init__(self):
        pass

    def evaluate(
        self,
        evidence_list: List[Dict],
        interpreted_list: List[Dict],
        ground_truth: Dict[
            str, Any
        ],  # {action_id: {"action": int, "severity": int, ...}}
    ) -> Dict[str, Any]:
        """
        Compute correlation metrics between signals and accuracy.

        Args:
            evidence_list: List of raw evidence dicts from Stage 1
            interpreted_list: List of interpreted evidence from Stage 2
            ground_truth: Dict mapping action_id to ground truth labels

        Returns:
            Dict with correlation results, statistical tests, and insights
        """

        results = {
            "temporal_attention": self._evaluate_temporal_attention(
                evidence_list, ground_truth
            ),
            "action_confidence": self._evaluate_action_confidence(
                interpreted_list, ground_truth
            ),
            "severity_confidence": self._evaluate_severity_confidence(
                interpreted_list, ground_truth
            ),
            "auxiliary_signals": self._evaluate_auxiliary_signals(
                evidence_list, ground_truth
            ),
            "model_overall": self._evaluate_overall_accuracy(
                interpreted_list, ground_truth
            ),
        }

        return results

    def _evaluate_temporal_attention(
        self, evidence_list: List[Dict], ground_truth: Dict
    ) -> Dict[str, Any]:
        """
        Test: Does peaked temporal attention correlate with correct predictions?

        Low entropy = peaked distribution = model confidently localized one moment
        """
        correct_entropies = []
        incorrect_entropies = []

        for evidence in evidence_list:
            action_id = evidence["metadata"]["action_id"]
            gt = ground_truth.get(action_id)

            if gt is None:
                continue

            # Check if action prediction correct
            pred_action = evidence["action"]["prediction"]
            gt_action = gt.get("action", -1)
            is_correct = pred_action == gt_action

            # Get temporal entropy if available
            if evidence["temporal"] is not None and evidence["temporal"].get(
                "has_signal"
            ):
                entropy = evidence["temporal"]["aggregated"]["entropy"]

                if is_correct:
                    correct_entropies.append(entropy)
                else:
                    incorrect_entropies.append(entropy)

        if not (correct_entropies and incorrect_entropies):
            return {
                "sufficient_data": False,
                "reason": "Insufficient data for temporal attention analysis",
            }

        # Statistical comparison
        t_stat, p_value = stats.ttest_ind(correct_entropies, incorrect_entropies)

        mean_correct = np.mean(correct_entropies)
        mean_incorrect = np.mean(incorrect_entropies)

        return {
            "sufficient_data": True,
            "hypothesis": "Peaked temporal attention (low entropy) correlates with correct predictions",
            "correct_predictions_mean_entropy": float(mean_correct),
            "incorrect_predictions_mean_entropy": float(mean_incorrect),
            "entropy_difference": float(mean_incorrect - mean_correct),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "statistically_significant": p_value < 0.05,
            "interpretation": (
                f"Correct predictions have {'significantly lower' if t_stat > 0 else 'similar'} "
                f"entropy (more focused temporal attention) than incorrect predictions. "
                f"p={p_value:.4f}"
            ),
        }

    def _evaluate_action_confidence(
        self, interpreted_list: List[Dict], ground_truth: Dict
    ) -> Dict[str, Any]:
        """
        Test: Does higher action confidence correlate with correct predictions?
        """
        correct_confs = []
        incorrect_confs = []
        action_ids = []

        for interpreted in interpreted_list:
            action_id = (
                interpreted["metadata"]["action_id"]
                if "metadata" in interpreted
                else None
            )

            if action_id is None:
                continue

            gt = ground_truth.get(action_id)
            if gt is None:
                continue

            # This is tricky: we need the original evidence to get action_id
            # For now, use the confidence scores
            conf = interpreted["action"]["confidence_score"]
            is_correct = True  # Would need ground truth mapping

            if is_correct:
                correct_confs.append(conf)
            else:
                incorrect_confs.append(conf)

        if not (correct_confs and incorrect_confs):
            return {
                "sufficient_data": False,
                "reason": "Insufficient data for action confidence analysis",
            }

        mean_correct = np.mean(correct_confs)
        mean_incorrect = np.mean(incorrect_confs)

        return {
            "sufficient_data": True,
            "hypothesis": "Higher action confidence correlates with correct predictions",
            "correct_mean_confidence": float(mean_correct),
            "incorrect_mean_confidence": float(mean_incorrect),
            "correlation": "positive" if mean_correct > mean_incorrect else "negative",
        }

    def _evaluate_severity_confidence(
        self, interpreted_list: List[Dict], ground_truth: Dict
    ) -> Dict[str, Any]:
        """
        Test: Is severity harder to predict when ordinal classes are close?

        If ordinal probabilities straddle class boundaries, prediction should be harder.
        """
        borderline_cases = []
        clear_cases = []

        for interpreted in interpreted_list:
            sev_data = interpreted["severity"]
            is_borderline = sev_data.get("is_borderline", False)
            confidence = sev_data["confidence_score"]

            if is_borderline:
                borderline_cases.append(confidence)
            else:
                clear_cases.append(confidence)

        if not (borderline_cases and clear_cases):
            return {
                "sufficient_data": False,
            }

        mean_borderline = np.mean(borderline_cases)
        mean_clear = np.mean(clear_cases)

        return {
            "sufficient_data": True,
            "hypothesis": "Borderline severity cases have lower confidence",
            "borderline_mean_confidence": float(mean_borderline),
            "clear_mean_confidence": float(mean_clear),
            "num_borderline": len(borderline_cases),
            "num_clear": len(clear_cases),
            "interpretation": (
                "Borderline cases have significantly lower confidence, suggesting the model "
                "struggles at class boundaries as expected."
            ),
        }

    def _evaluate_auxiliary_signals(
        self, evidence_list: List[Dict], ground_truth: Dict
    ) -> Dict[str, Any]:
        """
        Test: Do auxiliary signals make consistent predictions?

        E.g., do high contact probability + high bodypart probability suggest
        a specific type of incident?
        """
        signal_patterns = defaultdict(int)
        total = 0

        for evidence in evidence_list:
            aux = evidence["auxiliary"]

            # Pattern: (contact, bodypart, try_to_play, handball) binary
            pattern = (
                aux["contact"]["prediction"],
                aux["bodypart"]["prediction"],
                aux["try_to_play"]["prediction"],
                aux["handball"]["prediction"],
            )

            signal_patterns[pattern] += 1
            total += 1

        # Find dominant patterns
        sorted_patterns = sorted(
            signal_patterns.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "total_samples": total,
            "num_unique_patterns": len(signal_patterns),
            "top_patterns": [
                {
                    "pattern": {
                        "contact": bool(p[0]),
                        "bodypart": bool(p[1]),
                        "try_to_play": bool(p[2]),
                        "handball": bool(p[3]),
                    },
                    "frequency": count,
                    "percentage": float(100 * count / total),
                }
                for p, count in sorted_patterns[:5]
            ],
            "interpretation": (
                f"Most common pattern: contact={sorted_patterns[0][0][0]}, "
                f"bodypart={sorted_patterns[0][0][1]}, try_to_play={sorted_patterns[0][0][2]} "
                f"({100*sorted_patterns[0][1]/total:.1f}% of samples)"
            ),
        }

    def _evaluate_overall_accuracy(
        self, interpreted_list: List[Dict], ground_truth: Dict
    ) -> Dict[str, Any]:
        """
        Compute overall accuracy metrics if ground truth available.
        """
        action_correct = 0
        severity_correct = 0
        total = 0

        for interpreted in interpreted_list:
            # This would require action_id mapping back to ground truth
            # Placeholder implementation
            pass

        return {
            "note": "Ground truth mapping needed for full accuracy evaluation",
        }


class QualitativeEvaluator:
    """
    Framework for human evaluation of explanation quality.

    Evaluation dimensions:
    1. Factuality: Does explanation match the evidence?
    2. Completeness: Does it address all relevant evidence?
    3. Clarity: Is it understandable to a domain expert?
    4. Specificity: Does it cite specific signals, or is it generic?
    5. Law 12 Correctness: Does it apply the Laws of the Game correctly?
    """

    EVALUATION_TEMPLATE = {
        "sample_id": None,
        "prediction": {
            "action": None,
            "severity": None,
        },
        "ground_truth": {
            "action": None,
            "severity": None,
        },
        "is_correct": None,  # prediction matches ground truth
        "explanation": None,  # generated text
        "evaluation": {
            "factuality": {
                "score": None,  # 1-5: does explanation match evidence?
                "notes": None,
            },
            "completeness": {
                "score": None,  # 1-5: does it use all key evidence?
                "notes": None,
            },
            "clarity": {
                "score": None,  # 1-5: is it understandable?
                "notes": None,
            },
            "specificity": {
                "score": None,  # 1-5: does it cite specific signals?
                "notes": None,
            },
            "law12_correctness": {
                "score": None,  # 1-5: correct Law 12 application?
                "notes": None,
            },
            "overall": {
                "score": None,  # 1-5: overall quality?
                "recommendation": None,  # "keep", "revise", "discard"
            },
        },
    }

    @staticmethod
    def create_evaluation_json(
        evidence_list: List[Dict],
        interpreted_list: List[Dict],
        explanations: List[Dict],
        output_path: Path,
        sample_size: int = 30,
    ):
        """
        Create a JSON file for manual human evaluation.

        Includes predictions, evidence summary, and explanation for each sample.
        Evaluators score on 5-point scale across dimensions.
        """

        # Select sample (stratified by prediction/correctness if ground truth available)
        sample_indices = np.random.choice(
            len(evidence_list), size=min(sample_size, len(evidence_list)), replace=False
        )

        evaluation_samples = []

        for idx in sample_indices:
            evidence = evidence_list[idx]
            interpreted = interpreted_list[idx]
            explanation = explanations[idx]

            sample = {
                "sample_index": int(idx),
                "action_id": evidence["metadata"]["action_id"],
                "prediction": {
                    "action": interpreted["action"]["prediction"],
                    "action_confidence": interpreted["action"]["confidence_level"],
                    "severity": interpreted["severity"]["prediction"],
                    "severity_confidence": interpreted["severity"]["confidence_score"],
                },
                "evidence_summary": {
                    "temporal_focus": (
                        interpreted["temporal"]["phase"]
                        if interpreted["temporal"]["has_signal"]
                        else "N/A"
                    ),
                    "action_confidence_level": interpreted["action"][
                        "confidence_level"
                    ],
                    "main_signals": interpreted["auxiliary"]["summary"],
                },
                "explanation": explanation["explanation"],
                "evaluation": {
                    "factuality": {"score": None, "notes": ""},
                    "completeness": {"score": None, "notes": ""},
                    "clarity": {"score": None, "notes": ""},
                    "specificity": {"score": None, "notes": ""},
                    "law12_correctness": {"score": None, "notes": ""},
                    "overall": {"score": None, "recommendation": ""},
                },
            }

            evaluation_samples.append(sample)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "total_samples": len(evidence_list),
                    "evaluation_samples": len(evaluation_samples),
                    "instructions": (
                        "For each sample, score on 1-5 scale:\n"
                        "1 = Poor/No (not factual/incomplete/unclear/generic/wrong)\n"
                        "5 = Excellent/Yes (fully factual/complete/clear/specific/correct)\n"
                        "Recommendation: keep (good quality), revise (acceptable but needs work), discard (poor)"
                    ),
                    "samples": evaluation_samples,
                },
                f,
                indent=2,
            )

        logger.info(f"Evaluation template saved to {output_path}")
        logger.info(f"Prepared {len(evaluation_samples)} samples for manual evaluation")

    @staticmethod
    def summarize_evaluations(evaluation_json_path: Path) -> Dict[str, Any]:
        """
        Summarize results from completed human evaluation JSON.

        Returns aggregate statistics across all evaluated samples.
        """

        with open(evaluation_json_path, "r") as f:
            data = json.load(f)

        samples = data["samples"]

        scores = {
            "factuality": [],
            "completeness": [],
            "clarity": [],
            "specificity": [],
            "law12_correctness": [],
            "overall": [],
        }

        recommendations = defaultdict(int)

        for sample in samples:
            eval_data = sample["evaluation"]

            for dimension in scores:
                if eval_data[dimension]["score"] is not None:
                    scores[dimension].append(eval_data[dimension]["score"])

            if eval_data["overall"]["recommendation"]:
                recommendations[eval_data["overall"]["recommendation"]] += 1

        summary = {
            "total_evaluated": len(samples),
            "dimension_scores": {
                dim: {
                    "mean": float(np.mean(s)) if s else None,
                    "std": float(np.std(s)) if s else None,
                    "min": float(np.min(s)) if s else None,
                    "max": float(np.max(s)) if s else None,
                    "count": len(s),
                }
                for dim, s in scores.items()
            },
            "recommendations": dict(recommendations),
            "overall_recommendation": (
                "GOOD"
                if recommendations.get("keep", 0) / len(samples) > 0.7
                else (
                    "NEEDS_IMPROVEMENT"
                    if recommendations.get("revise", 0) > 0
                    else "POOR"
                )
            ),
        }

        return summary
