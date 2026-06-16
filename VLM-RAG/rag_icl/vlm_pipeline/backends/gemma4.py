"""
backends/gemma4.py
==================
Gemma 4 vision-language model backend: 12B and 31B.

Uses AutoModelForImageTextToText + AutoProcessor.
device_map="auto" so SLURM --gres=gpu:N controls GPU count.
Optional 4-bit / 8-bit quantization.

Thinking mode: Gemma4 has no native /think token. When enable_thinking=True
we prepend a chain-of-thought instruction asking the model to reason before
answering, then strip any <think>...</think> block from the output.
"""

import re
import torch
from typing import List, Optional
from PIL import Image

from ..prompts.templates import SYSTEM_PROMPT

_COT_INSTRUCTION = (
    "Before giving your final JSON answer, briefly reason about the force level, "
    "body part, and whether the action is aimed at the ball. "
    "Then output ONLY the JSON on the last line."
)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> if present, then return the remainder."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class Gemma4Backend:
    def __init__(
        self,
        model_name: str = "google/gemma-4-12b-it",
        enable_thinking: bool = False,
        quantize: str = "",
    ):
        from transformers import AutoProcessor

        self.enable_thinking = enable_thinking
        print(
            f"[Gemma4] Loading {model_name} "
            f"(thinking={'ON' if enable_thinking else 'OFF'}, quantize={quantize or 'none'})..."
        )

        model_kwargs: dict = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if quantize in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for quantized loading. "
                    "Install it with `pip install bitsandbytes`."
                ) from exc

            if quantize == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

        try:
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, **model_kwargs
            )
        except Exception:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        self.model.eval()
        print("[Gemma4] Ready.")

    def classify(
        self,
        frames_per_view: List[List[Image.Image]],
        prompt: str,
        extra_images: Optional[List[Image.Image]] = None,
    ) -> str:
        # Flatten frames to a single ordered list with view-label text interleaved
        content = []

        if extra_images:
            content.append(
                {"type": "text", "text": "[Reference examples from training data:]"}
            )
            for img in extra_images:
                content.append({"type": "image"})

        for v_idx, frames in enumerate(frames_per_view):
            label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            content.append({"type": "text", "text": f"\n[{label}]"})
            for _ in frames:
                content.append({"type": "image"})

        text_prompt = (
            f"\n\n{_COT_INSTRUCTION}\n\n{prompt}"
            if self.enable_thinking
            else f"\n\n{prompt}"
        )
        content.append({"type": "text", "text": text_prompt})

        messages = [
            {"role": "user", "content": content},
        ]

        # Collect all PIL images in the same order as {"type": "image"} entries
        all_images: List[Image.Image] = []
        if extra_images:
            all_images.extend(extra_images)
        for frames in frames_per_view:
            all_images.extend(frames)

        # Apply chat template to get text; processor handles image interleaving
        try:
            text_input = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text_input],
                images=all_images if all_images else None,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
        except Exception:
            # Fallback: some Gemma processor versions accept images= differently
            inputs = self.processor(
                text=[f"{SYSTEM_PROMPT}\n\n{prompt}"],
                images=all_images if all_images else None,
                return_tensors="pt",
            ).to(self.model.device)

        max_new_tokens = 1024 if self.enable_thinking else 512

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = out[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _strip_thinking(raw)
