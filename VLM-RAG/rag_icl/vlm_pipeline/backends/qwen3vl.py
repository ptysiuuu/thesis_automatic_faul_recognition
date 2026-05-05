"""
backends/qwen3vl.py
===================
Qwen3-VL-8B-Instruct backend.
Successor to Qwen2.5-VL — same vision encoder, updated language model.
API differs from Qwen2.5-VL: apply_chat_template handles everything,
no process_vision_info needed.
"""

import torch
from typing import List, Optional
from PIL import Image

from ..prompts.templates import SYSTEM_PROMPT


class Qwen3VLBackend:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        print(f"[Qwen3VL] Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True,
        )
        self.model.eval()
        print("[Qwen3VL] Ready.")

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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]

        # Qwen3-VL: apply_chat_template returns full inputs directly
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
            )

        generated_ids_trimmed = [
            out[len(inp):] for inp, out in
            zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
