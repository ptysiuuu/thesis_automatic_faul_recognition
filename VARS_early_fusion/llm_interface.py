"""
VARS Explainability: Stage 3 — LLM Explanation Generation

Generates natural language explanations using either:
1. Gemini API (recommended for quality, requires API key)
2. Local LLMs (Mistral, Llama, etc. via transformers/vLLM)
3. Anthropic Claude API (alternative premium option)

The LLM is given structured evidence and instructed to:
- Only refer to information in the evidence (never hallucinate visual details)
- Follow Law 12 framework for card decisions
- Be transparent about uncertainty
- Generate 3-5 sentence referee-style explanations
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    """Base class for LLM backends."""

    @abstractmethod
    def generate_explanation(self, evidence: Dict, interpreted: Dict) -> Dict[str, Any]:
        """
        Generate an explanation given evidence and interpretation.

        Returns:
            Dict with keys:
            - 'explanation': The generated text
            - 'confidence': Model's self-assessment of confidence
            - 'model': Which backend generated this
            - 'full_response': Raw response for debugging
        """
        pass


class GeminiInterface(LLMInterface):
    """
    Gemini API backend for explanation generation.

    Requires GOOGLE_API_KEY environment variable or api_key parameter.
    Uses google-generativeai library.
    """

    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.7):
        """
        Initialize Gemini interface.

        Args:
            api_key: Gemini API key (or uses GOOGLE_API_KEY env var)
            temperature: Sampling temperature for generation
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )

        self.genai = genai
        self.temperature = temperature

        # Get API key from parameter or environment
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not provided. "
                "Pass api_key parameter or set GOOGLE_API_KEY environment variable."
            )

        self.genai.configure(api_key=key)
        self.model = self.genai.GenerativeModel("gemini-3.1-flash-lite")
        logger.info("Initialized Gemini API interface")

    def generate_explanation(self, evidence: Dict, interpreted: Dict) -> Dict[str, Any]:
        """Generate explanation using Gemini API."""

        # Construct the prompt
        prompt = self._build_prompt(evidence, interpreted)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    temperature=self.temperature,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=300,
                ),
            )

            explanation_text = response.text

            return {
                "explanation": explanation_text,
                "confidence": "high",  # Gemini provides safety ratings; could parse those
                "model": "gemini-pro",
                "full_response": str(response),
            }

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "explanation": "Error generating explanation from Gemini API.",
                "confidence": "failed",
                "model": "gemini-pro",
                "error": str(e),
            }

    def _build_prompt(self, evidence: Dict, interpreted: Dict) -> str:
        """Construct the prompt for Gemini."""

        action_data = interpreted["action"]
        severity_data = interpreted["severity"]
        aux_data = interpreted["auxiliary"]
        temporal_data = interpreted["temporal"]
        confidence_data = interpreted["confidence"]

        action_name = action_data["prediction"]
        action_confidence = action_data["confidence_level"]

        severity_name = severity_data["prediction"]
        severity_reasoning = severity_data["ordinal_reasoning"]

        temporal_focus = ""
        if temporal_data["has_signal"]:
            temporal_focus = f"\n- Temporal Focus: {temporal_data['phase']} phase ({temporal_data['localization_strength']})"

        aux_signals = "\n".join(
            [f"  • {sig['signal']}: {sig['evidence']}" for sig in aux_data["signals"]]
        )

        prompt = f"""You are a football referee assistant explaining a VAR decision on foul recognition and misconduct (Law 12).

Based on the following structured evidence from a multi-view video analysis model, generate a concise referee-style explanation of the decision (3-4 sentences).

CRITICAL RULES:
1. Only refer to information explicitly provided below. Never invent visual observations.
2. Use the Law 12 framework: careless (no card) → reckless (yellow) → excessive force (red) → violent conduct (red)
3. Acknowledge uncertainty if the model's confidence is low.
4. Be direct and speak in a referee's voice.
5. Do not say "the model thinks" - speak as if you are the referee explaining your decision.

PREDICTION SUMMARY:
- Action: {action_name} (Confidence: {action_confidence})
- Severity: {severity_name}

STEP-BY-STEP SEVERITY REASONING:
{severity_reasoning}

TEMPORAL ANALYSIS:
{temporal_focus}

PHYSICAL INDICATORS:
{aux_signals}

OVERALL MODEL CONFIDENCE:
- {confidence_data['overall_text']}
- Main uncertainties: {', '.join(confidence_data['main_uncertainties'])}

CONFUSION (if any):
- {action_data.get('confusion', 'None')}

Now generate the explanation. Be concise, direct, and ground your decision in the evidence above."""

        return prompt


class AnthropicInterface(LLMInterface):
    """
    Anthropic Claude API backend for explanation generation.

    Requires ANTHROPIC_API_KEY environment variable.
    Uses anthropic library.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
    ):
        """
        Initialize Anthropic interface.

        Args:
            api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
            model: Claude model to use
            temperature: Sampling temperature
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic not installed. " "Install with: pip install anthropic"
            )

        self.Anthropic = Anthropic
        self.model = model
        self.temperature = temperature

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Pass api_key parameter or set ANTHROPIC_API_KEY environment variable."
            )

        self.client = Anthropic(api_key=key)
        logger.info(f"Initialized Anthropic interface (model: {model})")

    def generate_explanation(self, evidence: Dict, interpreted: Dict) -> Dict[str, Any]:
        """Generate explanation using Anthropic Claude API."""

        prompt = self._build_prompt(evidence, interpreted)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            explanation_text = response.content[0].text

            return {
                "explanation": explanation_text,
                "confidence": "high",
                "model": self.model,
                "stop_reason": response.stop_reason,
            }

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {
                "explanation": "Error generating explanation from Anthropic API.",
                "confidence": "failed",
                "model": self.model,
                "error": str(e),
            }

    def _build_prompt(self, evidence: Dict, interpreted: Dict) -> str:
        """Construct the prompt for Claude."""

        # Same structure as Gemini
        action_data = interpreted["action"]
        severity_data = interpreted["severity"]
        aux_data = interpreted["auxiliary"]
        temporal_data = interpreted["temporal"]
        confidence_data = interpreted["confidence"]

        action_name = action_data["prediction"]
        action_confidence = action_data["confidence_level"]
        severity_name = severity_data["prediction"]
        severity_reasoning = severity_data["ordinal_reasoning"]

        temporal_focus = ""
        if temporal_data["has_signal"]:
            temporal_focus = f"\n- Temporal Focus: {temporal_data['phase']} phase ({temporal_data['localization_strength']})"

        aux_signals = "\n".join(
            [f"  • {sig['signal']}: {sig['evidence']}" for sig in aux_data["signals"]]
        )

        prompt = f"""You are a football referee assistant explaining a VAR decision on foul recognition and misconduct (Law 12).

Based on the following structured evidence from a multi-view video analysis model, generate a concise referee-style explanation of the decision (3-4 sentences).

CRITICAL RULES:
1. Only refer to information explicitly provided below. Never invent visual observations.
2. Use the Law 12 framework: careless (no card) → reckless (yellow) → excessive force (red) → violent conduct (red)
3. Acknowledge uncertainty if the model's confidence is low.
4. Be direct and speak in a referee's voice.
5. Do not say "the model thinks" - speak as if you are the referee explaining your decision.

PREDICTION SUMMARY:
- Action: {action_name} (Confidence: {action_confidence})
- Severity: {severity_name}

STEP-BY-STEP SEVERITY REASONING:
{severity_reasoning}

TEMPORAL ANALYSIS:
{temporal_focus}

PHYSICAL INDICATORS:
{aux_signals}

OVERALL MODEL CONFIDENCE:
- {confidence_data['overall_text']}
- Main uncertainties: {', '.join(confidence_data['main_uncertainties'])}

CONFUSION (if any):
- {action_data.get('confusion', 'None')}

Now generate the explanation. Be concise, direct, and ground your decision in the evidence above."""

        return prompt


class LocalLLMInterface(LLMInterface):
    """
    Local LLM backend using transformers or ollama.

    Supports models like Mistral-7B-Instruct, Llama-2-7B-Chat, etc.
    Requires sufficient GPU memory.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.7,
    ):
        """
        Initialize local LLM interface.

        Args:
            model_name: HuggingFace model ID or local path
            temperature: Sampling temperature
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading local model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        logger.info(f"Model loaded on device: {self.device}")

    def generate_explanation(self, evidence: Dict, interpreted: Dict) -> Dict[str, Any]:
        """Generate explanation using local LLM."""

        prompt = self._build_prompt(evidence, interpreted)

        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=True,
                )

            explanation_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            # Extract only the generated part (after the prompt)
            if prompt in explanation_text:
                explanation_text = explanation_text.split(prompt)[-1].strip()

            return {
                "explanation": explanation_text,
                "confidence": "moderate",  # Local models' confidence is harder to estimate
                "model": self.model_name,
            }

        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            return {
                "explanation": "Error generating explanation from local LLM.",
                "confidence": "failed",
                "model": self.model_name,
                "error": str(e),
            }

    def _build_prompt(self, evidence: Dict, interpreted: Dict) -> str:
        """Construct the prompt for local LLM."""

        # Same structure as APIs
        action_data = interpreted["action"]
        severity_data = interpreted["severity"]
        aux_data = interpreted["auxiliary"]
        temporal_data = interpreted["temporal"]
        confidence_data = interpreted["confidence"]

        action_name = action_data["prediction"]
        action_confidence = action_data["confidence_level"]
        severity_name = severity_data["prediction"]
        severity_reasoning = severity_data["ordinal_reasoning"]

        temporal_focus = ""
        if temporal_data["has_signal"]:
            temporal_focus = f"- Temporal Focus: {temporal_data['phase']} phase ({temporal_data['localization_strength']})"

        aux_signals = "\n".join(
            [f"• {sig['signal']}: {sig['evidence']}" for sig in aux_data["signals"]]
        )

        system_prompt = """You are a football referee assistant. Explain a VAR decision on foul recognition (Law 12).

RULES:
1. Only use the provided evidence - never invent observations.
2. Use Law 12 framework: careless (no card) → reckless (yellow) → excessive (red).
3. Acknowledge uncertainty if confidence is low.
4. Be direct, speak as the referee, not as "the model".
5. Keep to 3-4 sentences.

"""

        evidence_block = f"""ACTION: {action_name} ({action_confidence} confidence)
SEVERITY: {severity_name}

REASONING:
{severity_reasoning}

ANALYSIS:
{temporal_focus}

SIGNALS:
{aux_signals}

CONFIDENCE: {confidence_data['overall_level']}
"""

        return system_prompt + evidence_block + "\nExplanation:"


# ============================================================================
# Factory function
# ============================================================================


def get_llm_interface(
    backend: str = "gemini",
    model: str = "",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
) -> LLMInterface:
    """
    Get an LLM interface instance.

    Args:
        backend: One of 'gemini', 'anthropic', 'local'
        model: Model name (required for 'local', optional for others)
        api_key: API key for cloud backends
        temperature: Sampling temperature

    Returns:
        LLMInterface instance
    """

    if backend == "gemini":
        return GeminiInterface(api_key=api_key, temperature=temperature)

    elif backend == "anthropic":
        return AnthropicInterface(api_key=api_key, temperature=temperature)

    elif backend == "local":
        if not model:
            model = "mistralai/Mistral-7B-Instruct-v0.2"
        return LocalLLMInterface(model_name=model, temperature=temperature)

    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
