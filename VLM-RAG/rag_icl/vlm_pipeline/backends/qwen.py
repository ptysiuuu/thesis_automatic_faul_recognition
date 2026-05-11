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
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        quantize_4bit: bool = False,
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"[QwenVL] Loading {model_name}...")

        if model_name in PROCESSOR_FALLBACK:
            print(f"[QwenVL] Processor fallback: loading from {BASE_PROCESSOR}")
            processor_source = BASE_PROCESSOR
        else:
            processor_source = model_name

        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        if quantize_4bit:
            from transformers import BitsAndBytesConfig

            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for 4-bit Qwen loading. "
                    "Install it with `pip install bitsandbytes`."
                ) from exc

                if not hasattr(torch.nn.Module, "set_submodule"):

                    def _set_submodule(self, target: str, module):
                        parent_path, _, child_name = target.rpartition(".")
                        parent = (
                            self.get_submodule(parent_path) if parent_path else self
                        )
                        setattr(parent, child_name, module)

                    torch.nn.Module.set_submodule = _set_submodule

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.processor = AutoProcessor.from_pretrained(
            processor_source, trust_remote_code=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.model.eval()
        print("[QwenVL] Ready.")

    def classify(
        self,
        frames_per_view: List[List[Image.Image]],
        prompt: str,
        extra_images: Optional[List[Image.Image]] = None,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        content = []
        if extra_images:
            content.append(
                {"type": "text", "text": "[Reference examples from training data:]"}
            )
            for img in extra_images:
                content.append({"type": "image", "image": img})

        for v_idx, frames in enumerate(frames_per_view):
            label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            content.append({"type": "text", "text": f"\n[{label}]"})
            content.append({"type": "video", "video": frames})
        content.append({"type": "text", "text": f"\n\n{prompt}"})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        video_kwargs = video_kwargs or {}
        inputs = self.processor(
            text=[text_input],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to("cuda")

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]
