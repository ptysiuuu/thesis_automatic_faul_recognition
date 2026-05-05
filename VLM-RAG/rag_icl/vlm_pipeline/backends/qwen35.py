"""
backends/qwen35.py
==================
Qwen3.5-9B — reasoning VLM that thinks before answering.
The model produces long <think>...</think> chains then outputs JSON.
Key fix: use enable_thinking=False to suppress the chain for speed,
OR increase max_new_tokens and parse the final JSON from the think output.
We default to enable_thinking=False for reliability.
"""

import re
import torch
from typing import List, Optional
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from ..prompts.templates import SYSTEM_PROMPT


def _extract_final_json(text: str) -> str:
    """Extract JSON from output that may contain a think block."""
    # Strip <think>...</think>
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "{" in clean:
        return clean
    # Fallback: last JSON object anywhere in text
    matches = list(re.finditer(r"\{[^{}]+\}", text, re.DOTALL))
    if matches:
        return matches[-1].group()
    return text


class Qwen35Backend:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-9B",
                 enable_thinking: bool = False):
        print(f"[Qwen3.5] Loading {model_name} (thinking={'ON' if enable_thinking else 'OFF'})...")
        self.enable_thinking = enable_thinking
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True,
        )
        self.model.eval()
        print("[Qwen3.5] Ready.")

    def classify(self, frames_per_view: List[List[Image.Image]],
                 prompt: str,
                 extra_images: Optional[List[Image.Image]] = None) -> str:
        content = []
        if extra_images:
            content.append({"type": "text",
                             "text": "[Reference examples from training data:]"})
            for img in extra_images:
                content.append({"type": "image", "image": img})

        for v_idx, frames in enumerate(frames_per_view):
            label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            content.append({"type": "text", "text": f"\n[{label}]"})
            for frame in frames:
                content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": f"\n\n{prompt}"})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=self.enable_thinking,
        ).to(self.model.device)

        max_tokens = 4096 if self.enable_thinking else 512
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        raw = self.processor.decode(generated_ids, skip_special_tokens=True)
        return _extract_final_json(raw)
