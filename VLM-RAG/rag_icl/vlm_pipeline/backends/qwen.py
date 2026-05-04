"""
backends/qwen.py
================
Qwen2.5-VL-7B-Instruct backend.
Also handles Qwen2.5-VL finetunes with broken processor configs
(e.g. Video-R1) via the PROCESSOR_FALLBACK mechanism.
"""

import torch
from typing import List, Optional
from PIL import Image

from ..prompts.templates import SYSTEM_PROMPT

# Models that are Qwen2.5-VL finetunes but ship without a valid
# preprocessor_config.json. Load processor from base model instead.
PROCESSOR_FALLBACK = {
    "Video-R1/Video-R1-7B",
    "Video-R1/Qwen2.5-VL-7B-COT-SFT",
}
BASE_PROCESSOR = "Qwen/Qwen2.5-VL-7B-Instruct"


class QwenVLBackend:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"[QwenVL] Loading {model_name}...")

        if model_name in PROCESSOR_FALLBACK:
            print(f"[QwenVL] Processor fallback: loading from {BASE_PROCESSOR}")
            processor_source = BASE_PROCESSOR
        else:
            processor_source = model_name

        self.processor = AutoProcessor.from_pretrained(
            processor_source, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True,
        )
        self.model.eval()
        print("[QwenVL] Ready.")

    def classify(self, frames_per_view: List[List[Image.Image]],
                 prompt: str,
                 extra_images: Optional[List[Image.Image]] = None) -> str:
        from qwen_vl_utils import process_vision_info

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
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input], images=img_inputs, videos=vid_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=512,
                do_sample=False, temperature=None, top_p=None,
            )
        generated = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]
