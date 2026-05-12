"""
VARS Explainability: Stage 2 — Evidence Interpretation

Converts raw model signals into structured, semantically meaningful evidence.
No learning involved — pure logic that encodes domain knowledge about:
- Temporal attention patterns (what they imply about the incident)
- Action confidence (degree of certainty)
- Severity ordinal interpretation (step-by-step what each cumulative threshold means)
- Auxiliary signal meaning (how contact/bodypart/try-to-play map to Laws of the Game)
- View reliability (which cameras were most informative)
"""

from typing import Dict, List, Any
from config.classes import INVERSE_EVENT_DICTIONARY


class EvidenceInterpreter:
    """
    Maps raw model signals to domain-meaningful structured text.

    Encoding domain knowledge about:
    - Law 12 (misconduct): careless (no card) → reckless (yellow) → excessive force (red)
    - Temporal patterns: approach phase, contact moment, aftermath reaction
    - Physical indicators: contact type (bodypart), intent (try-to-play), special case (handball)
    - Confidence metrics: gap between top predictions, entropy of distributions
    """

    # Law 12 framework
    SEVERITY_NAMES = {
        0: "No card",
        1: "Yellow card (reckless)",
        2: "Red card (excessive force)",
        3: "Red card (violent conduct)",
    }

    # Temporal phases
    TEMPORAL_PHASES = {
        "approach": (0, 2),  # tokens 0-2: player moving toward opponent
        "contact": (3, 5),  # tokens 3-5: contact moment
        "aftermath": (6, 7),  # tokens 6-7: aftermath (fall, reaction)
    }

    # Action context for severity
    ACTION_SEVERITY_CONTEXT = {
        0: "Foul committed",  # Foul
        1: "Advantage played",  # Advantage
        2: "Ball out of play",  # Ball Out
        3: "Offside",  # Offside
        4: "Player off ball",  # Player Off
        5: "Ball touch",  # Ball Touch
        6: "Temp_Goal",  # Temp_Goal (special)
        7: "Direct Red",  # Direct Red
    }

    def __init__(self):
        pass

    def interpret(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret all signals from the evidence dictionary.

        Returns a structured interpretation with:
        - Temporal focus (where in the clip the model focused and what it suggests)
        - Action assessment (what action, how confident, what confused it if anything)
        - Severity reasoning (ordinal step-by-step interpretation)
        - Physical indicators (what auxiliary signals say about the incident)
        - Confidence assessment (overall model certainty and uncertainty)
        - Camera reliability (which views were most informative)
        """

        action_interp = self._interpret_action(evidence["action"])
        severity_interp = self._interpret_severity(
            evidence["severity"], evidence["action"]
        )
        auxiliary_interp = self._interpret_auxiliary(
            evidence["auxiliary"], evidence["action"]
        )
        temporal_interp = self._interpret_temporal(evidence["temporal"])
        confidence_interp = self._interpret_confidence(
            evidence["action"], evidence["severity"]
        )

        return {
            "action": action_interp,
            "severity": severity_interp,
            "auxiliary": auxiliary_interp,
            "temporal": temporal_interp,
            "confidence": confidence_interp,
            "summary": self._generate_summary(
                action_interp,
                severity_interp,
                auxiliary_interp,
                temporal_interp,
                confidence_interp,
            ),
        }

    def _interpret_action(self, action_data: Dict) -> Dict[str, Any]:
        """
        Interpret action classification signals.

        Answers: What action? How confident? What was the confusion?
        """
        pred = action_data["prediction"]
        pred_name = action_data["prediction_name"]
        confidence = action_data["confidence"]
        top2_probs = action_data["top2"]["probs"]
        top2_names = action_data["top2"]["names"]
        confidence_gap = action_data["top2"]["confidence_gap"]

        # Confidence level interpretation
        if confidence > 0.8:
            confidence_level = "HIGH"
            confidence_text = "The model is very confident in this prediction."
        elif confidence > 0.6:
            confidence_level = "MODERATE"
            confidence_text = "The model is moderately confident in this prediction."
        elif confidence > 0.4:
            confidence_level = "LOW"
            confidence_text = "The model is uncertain; the incident could be interpreted as one of several actions."
        else:
            confidence_level = "VERY_LOW"
            confidence_text = "The model is highly uncertain between multiple actions."

        # Confusion analysis
        confusion_text = ""
        if confidence_gap < 0.1:
            # Very close between top 2
            confusion_text = (
                f"The model struggled to distinguish between {top2_names[0]} "
                f"({top2_probs[0]:.1%}) and {top2_names[1]} ({top2_probs[1]:.1%}). "
                f"These actions likely share similar visual characteristics."
            )
        elif confidence_gap < 0.2:
            confusion_text = (
                f"The model leaned toward {top2_names[0]} but was somewhat uncertain. "
                f"The second candidate was {top2_names[1]} ({top2_probs[1]:.1%})."
            )
        elif confidence_gap < 0.4:
            confusion_text = (
                f"The main alternative interpretation was {top2_names[1]} ({top2_probs[1]:.1%}), "
                f"but the model clearly favored {top2_names[0]}."
            )

        return {
            "prediction": pred_name,
            "confidence_score": float(confidence),
            "confidence_level": confidence_level,
            "confidence_text": confidence_text,
            "confusion": (
                confusion_text
                if confusion_text
                else "No significant confusion detected."
            ),
            "top2_alternatives": [
                {
                    "action": top2_names[0],
                    "probability": float(top2_probs[0]),
                },
                {
                    "action": top2_names[1],
                    "probability": float(top2_probs[1]),
                },
            ],
        }

    def _interpret_severity(
        self, severity_data: Dict, action_data: Dict
    ) -> Dict[str, Any]:
        """
        Interpret severity (ordinal regression) signals.

        The ordinal model predicts: P(severity > 0), P(severity > 1), P(severity > 2)
        where 0=no card, 1=yellow, 2=red (excess force), 3=red (violent)

        Answers: What card? What is the step-by-step reasoning?
        """
        pred = severity_data["prediction"]
        ordinal_probs = severity_data["ordinal_probs"]  # [P(>0), P(>1), P(>2)]
        class_probs = severity_data["class_probs"]  # [P(0), P(1), P(2), P(3)]

        severity_name = self.SEVERITY_NAMES.get(pred, "Unknown")

        # Step-by-step ordinal interpretation
        ordinal_text = self._interpret_ordinal_steps(ordinal_probs)

        # Confidence of severity
        severity_confidence = class_probs[pred]

        # Is this borderline?
        is_borderline = False
        borderline_text = ""

        # Check for boundaries
        if pred == 0 and ordinal_probs[0] < 0.7:
            is_borderline = True
            borderline_text = (
                f"Borderline: The model is not entirely sure this is a foul. "
                f"It gives {ordinal_probs[0]:.1%} confidence that it's at least careless."
            )
        elif pred == 1 and (abs(ordinal_probs[1] - 0.5) < 0.15):
            is_borderline = True
            borderline_text = (
                f"Borderline yellow/red: The model is uncertain between yellow and red card. "
                f"P(reckless): {ordinal_probs[1]:.1%}, P(excessive): {ordinal_probs[2]:.1%}."
            )
        elif pred == 2 and ordinal_probs[1] < 0.7:
            is_borderline = True
            borderline_text = (
                f"Clear red card: The model is confident this exceeds recklessness. "
                f"P(excessive force): {ordinal_probs[2]:.1%}."
            )

        return {
            "prediction": severity_name,
            "prediction_code": int(pred),
            "confidence_score": float(severity_confidence),
            "ordinal_reasoning": ordinal_text,
            "is_borderline": is_borderline,
            "borderline_note": borderline_text if borderline_text else "",
            "ordinal_probabilities": {
                "foul_or_not": float(ordinal_probs[0]),
                "yellow_or_worse": float(ordinal_probs[1]),
                "red_or_worse": float(ordinal_probs[2]),
            },
            "class_probabilities": {
                "no_card": float(class_probs[0]),
                "yellow_card": float(class_probs[1]),
                "red_card_excess_force": float(class_probs[2]),
                "red_card_violent": float(class_probs[3]),
            },
        }

    def _interpret_ordinal_steps(self, ordinal_probs: List[float]) -> str:
        """
        Convert ordinal probabilities to step-by-step natural language.

        ordinal_probs: [P(severity > 0), P(severity > 1), P(severity > 2)]
        """
        p_foul = ordinal_probs[0]
        p_yellow = ordinal_probs[1]
        p_red = ordinal_probs[2]

        steps = []

        # Step 1: Is it a foul?
        if p_foul > 0.8:
            steps.append(f"✓ Definitely a foul ({p_foul:.1%} confidence)")
        elif p_foul > 0.5:
            steps.append(f"✓ Likely a foul ({p_foul:.1%} confidence)")
        else:
            steps.append(
                f"✗ Probably not a foul ({1-p_foul:.1%} confidence this is play-on)"
            )
            return " → ".join(steps)

        # Step 2: Reckless or worse?
        if p_yellow > 0.8:
            steps.append(f"→ Reckless play, yellow card ({p_yellow:.1%} confidence)")
        elif p_yellow > 0.5:
            steps.append(
                f"→ Possibly reckless, potential yellow ({p_yellow:.1%} confidence)"
            )
        else:
            steps.append(f"→ Not reckless, just careless (no card)")
            return " → ".join(steps)

        # Step 3: Excessive force or worse?
        if p_red > 0.8:
            steps.append(
                f"→ Excessive force or violent conduct, red card ({p_red:.1%} confidence)"
            )
        elif p_red > 0.5:
            steps.append(
                f"→ Potentially excessive, might warrant red ({p_red:.1%} confidence)"
            )
        else:
            steps.append(f"→ Yellow card decision (not excessive force)")

        return " → ".join(steps)

    def _interpret_auxiliary(
        self, auxiliary_data: Dict, action_data: Dict
    ) -> Dict[str, Any]:
        """
        Interpret auxiliary signals and their Law 12 implications.

        These map directly to referee decision criteria:
        - contact: confirms a foul occurred (physical contact)
        - bodypart: location of contact (determines severity in some cases)
        - try_to_play: player attempted to play the ball (mitigating factor, reduces severity)
        - handball: separate special case (handball → different consequences)
        """
        contact = auxiliary_data["contact"]
        bodypart = auxiliary_data["bodypart"]
        try_to_play = auxiliary_data["try_to_play"]
        handball = auxiliary_data["handball"]

        signals = []

        # Contact signal
        if contact["probability"] > 0.7:
            signals.append(
                {
                    "signal": "Contact confirmed",
                    "evidence": f"Physical contact detected ({contact['probability']:.1%} confidence)",
                    "implication": "This is definitely a foul, not a simulation or professional foul only.",
                }
            )
        else:
            signals.append(
                {
                    "signal": "Contact uncertain",
                    "evidence": f"Weak contact signal ({contact['probability']:.1%} confidence)",
                    "implication": "Could be a professional foul or simulation.",
                }
            )

        # Bodypart signal (upper body vs lower body)
        if bodypart["probability"] > 0.6:
            signals.append(
                {
                    "signal": "Upper body contact",
                    "evidence": f"Head/arm/torso contact detected ({bodypart['probability']:.1%} confidence)",
                    "implication": "Contact to head significantly increases severity (potential red). Contact to arm/torso is context-dependent.",
                }
            )
        else:
            signals.append(
                {
                    "signal": "Lower body contact",
                    "evidence": f"Leg/foot contact likely ({1-bodypart['probability']:.1%} confidence)",
                    "implication": "Typically less severe than upper body contact.",
                }
            )

        # Try-to-play signal (mitigating factor)
        if try_to_play["probability"] > 0.6:
            signals.append(
                {
                    "signal": "Attempted to play the ball",
                    "evidence": f"Try-to-play detected ({try_to_play['probability']:.1%} confidence)",
                    "implication": "MITIGATING FACTOR: Player was not deliberately fouling; contact was incidental to attempting to play. Reduces severity.",
                }
            )
        else:
            signals.append(
                {
                    "signal": "Deliberate foul likely",
                    "evidence": f"No clear ball-play attempt ({1-try_to_play['probability']:.1%} confidence)",
                    "implication": "The foul appears intentional, not incidental. This increases severity.",
                }
            )

        # Handball signal (special case)
        if handball["probability"] > 0.6:
            signals.append(
                {
                    "signal": "Handball detected",
                    "evidence": f"Handball signal ({handball['probability']:.1%} confidence)",
                    "implication": "SPECIAL CASE: Handball fouls have different consequences (penalty vs direct free kick) and are separate from violent conduct considerations.",
                }
            )

        return {
            "signals": signals,
            "summary": self._summarize_auxiliary_signals(signals),
        }

    def _summarize_auxiliary_signals(self, signals: List[Dict]) -> str:
        """Generate a natural language summary of auxiliary signals."""
        key_signals = []

        for sig in signals:
            if "MITIGATING" in sig["implication"]:
                key_signals.append(f"Mitigating: {sig['signal']}")
            elif "SPECIAL CASE" in sig["implication"]:
                key_signals.append(f"Special case: {sig['signal']}")
            elif sig["probability"] > 0.7 if "probability" in sig else False:
                key_signals.append(sig["signal"])

        if not key_signals:
            return "No distinctive auxiliary signals detected."

        return " | ".join(key_signals)

    def _interpret_temporal(self, temporal_data: Dict) -> Dict[str, Any]:
        """
        Interpret temporal attention patterns.

        Answers: Where in the clip did the model focus? What does that tell us about the incident?
        """
        if temporal_data is None or not temporal_data.get("has_signal"):
            return {
                "has_signal": False,
                "reason": "No temporal attention data available (early fusion mode or other reason)",
                "focus_assessment": "",
            }

        agg = temporal_data["aggregated"]
        peak_token = agg["peak_token"]
        entropy = agg["entropy"]
        com = agg["center_of_mass"]

        # Determine temporal phase
        phase_name = self._classify_temporal_phase(peak_token)

        # Entropy interpretation
        if entropy < 2.0:
            localization = "sharply focused"
            localization_text = (
                f"The model clearly identified one moment as decisive (entropy: {entropy:.2f}). "
                f"This suggests confident localization of the contact or impact."
            )
        elif entropy < 3.0:
            localization = "moderately focused"
            localization_text = (
                f"The model focused on a specific window but not a single frame (entropy: {entropy:.2f}). "
                f"This is typical when contact and immediate aftermath are both important."
            )
        else:
            localization = "spread across frames"
            localization_text = (
                f"The model distributed attention across multiple frames (entropy: {entropy:.2f}). "
                f"This suggests the temporal signal was ambiguous or the incident unfolded gradually."
            )

        # Temporal phase implications
        if phase_name == "approach":
            phase_text = (
                "The model focused on the APPROACH phase (how the player moved toward the opponent). "
                "This typically indicates attention to the speed/force of entry, suggesting assessment "
                "of whether the challenge was reckless or excessive."
            )
        elif phase_name == "contact":
            phase_text = (
                "The model focused on the CONTACT moment (when physical contact occurred). "
                "This is the standard focus point and indicates the model properly identified when "
                "the foul occurred."
            )
        elif phase_name == "aftermath":
            phase_text = (
                "The model focused on the AFTERMATH phase (fall, reaction, recovery). "
                "This can indicate: (1) the model identified the impact's severity from the opponent's "
                "reaction, or (2) the approach/contact phases were ambiguous so the consequence was more "
                "informative."
            )
        else:
            phase_text = "Temporal phase unclear."

        # View consistency (if available)
        view_consistency_text = ""
        if "per_view" in temporal_data:
            view_entropies = [v["entropy"] for v in temporal_data["per_view"]]
            entropy_std = (
                sum(
                    (e - sum(view_entropies) / len(view_entropies)) ** 2
                    for e in view_entropies
                )
                / len(view_entropies)
            ) ** 0.5

            if entropy_std < 1.0:
                view_consistency_text = (
                    "All camera angles agreed on the temporal focus (consistent across views). "
                    "This indicates a strong, unambiguous signal."
                )
            else:
                view_consistency_text = (
                    "Different camera angles focused on different moments. "
                    "This suggests the temporal signal varies by viewpoint, which is common for complex incidents."
                )

        return {
            "has_signal": True,
            "peak_token": int(peak_token),
            "phase": phase_name,
            "entropy": float(entropy),
            "center_of_mass": float(com),
            "localization_strength": localization,
            "localization_text": localization_text,
            "phase_implications": phase_text,
            "view_consistency": view_consistency_text,
            "frame_mapping": temporal_data.get("token_frame_mapping", {}),
        }

    def _classify_temporal_phase(self, peak_token: int) -> str:
        """Classify which temporal phase the peak token falls into."""
        if peak_token <= 2:
            return "approach"
        elif peak_token <= 5:
            return "contact"
        else:
            return "aftermath"

    def _interpret_confidence(
        self, action_data: Dict, severity_data: Dict
    ) -> Dict[str, Any]:
        """
        Overall confidence assessment combining action and severity.

        Low confidence on either dimension signals uncertainty in the prediction.
        """
        action_conf = action_data["confidence"]
        action_gap = action_data["top2"]["confidence_gap"]
        severity_conf = severity_data["confidence"]

        overall_conf = (action_conf + severity_conf) / 2

        # Determine overall confidence level
        if overall_conf > 0.75:
            overall_level = "HIGH"
            overall_text = "The model is confident in both action and severity."
        elif overall_conf > 0.6:
            overall_level = "MODERATE"
            overall_text = "The model has moderate confidence, with some uncertainty on one dimension."
        elif overall_conf > 0.45:
            overall_level = "LOW"
            overall_text = "The model is uncertain; both action and severity have moderate confidence."
        else:
            overall_level = "VERY_LOW"
            overall_text = "The model is highly uncertain on both dimensions. This prediction should be treated with caution."

        # Specific uncertainties
        uncertainties = []
        if action_gap < 0.15:
            uncertainties.append(
                "Action is ambiguous (top 2 predictions are very close)"
            )
        if severity_conf < 0.5:
            uncertainties.append("Severity class boundaries are unclear")

        return {
            "overall_confidence": float(overall_conf),
            "overall_level": overall_level,
            "overall_text": overall_text,
            "action_confidence": float(action_conf),
            "action_confidence_gap": float(action_gap),
            "severity_confidence": float(severity_conf),
            "main_uncertainties": uncertainties if uncertainties else ["None"],
        }

    def _generate_summary(
        self,
        action_interp: Dict,
        severity_interp: Dict,
        auxiliary_interp: Dict,
        temporal_interp: Dict,
        confidence_interp: Dict,
    ) -> str:
        """
        Generate a concise plain-English summary of the interpretation.

        This is a stepping stone to the LLM explanation but comprehensible on its own.
        """
        parts = []

        # Action
        parts.append(
            f"Action: {action_interp['prediction']} ({action_interp['confidence_level']})"
        )

        # Severity
        parts.append(f"Severity: {severity_interp['prediction']}")

        # Temporal focus
        if temporal_interp["has_signal"]:
            parts.append(f"Focus: {temporal_interp['phase']} phase")

        # Key auxiliary signals
        aux_summary = auxiliary_interp.get("summary", "")
        if aux_summary and aux_summary != "No distinctive auxiliary signals detected.":
            parts.append(f"Signals: {aux_summary}")

        # Confidence caveat
        if confidence_interp["overall_level"] in ("LOW", "VERY_LOW"):
            parts.append(f"Confidence: {confidence_interp['overall_level']}")

        return " | ".join(parts)
