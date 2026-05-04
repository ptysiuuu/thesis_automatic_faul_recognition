
import torch
from typing import List, Optional
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from ..prompts.templates import SYSTEM_PROMPT

class Qwen35Backend:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-9B"):
        print(f"[Qwen3.5] Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
        self.model.eval()
        print("[Qwen3.5] Ready.")

    def classify(self, frames_per_view: List[List[Image.Image]], prompt: str, extra_images: Optional[List[Image.Image]] = None) -> str:
        content = []
        if extra_images:
            content.append({"type": "text", "text": "[Reference examples from training data:]"})
            for img in extra_images:
                # The AutoProcessor natively handles PIL Image objects!
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

        # Mirroring your exact snippet configuration
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated_ids, skip_special_tokens=True)
