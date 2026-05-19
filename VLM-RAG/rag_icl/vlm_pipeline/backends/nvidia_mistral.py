"""
backends/nvidia_mistral.py
==========================
NVIDIA NIM Mistral backend (OpenAI-compatible chat/completions API).
Uses base64-encoded images via data URI in image_url.
"""

import base64
import os
import re
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image

from ..prompts.templates import SYSTEM_PROMPT


def _pil_to_b64(img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class NvidiaMistralBackend:
    """
    Calls NVIDIA NIM chat/completions endpoint for Mistral models.

    Parameters
    ----------
    model_name : str
        NVIDIA model ID, e.g. "mistralai/mistral-large-3-675b-instruct-2512".
    api_key : str
        NVIDIA API key (or set NVIDIA_API_KEY).
    base_url : str
        Endpoint URL, default NVIDIA NIM integrate URL.
    timeout : int
        Request timeout in seconds.
    max_images_per_request : int
        Cap total images per request to avoid oversized payloads.
    image_quality : int
        JPEG quality for base64 encoding.
    """

    def __init__(
        self,
        model_name: str = "mistralai/mistral-large-3-675b-instruct-2512",
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1/chat/completions",
        timeout: int = 120,
        max_images_per_request: int = 16,
        image_quality: int = 85,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key not provided. Set NVIDIA_API_KEY or pass api_key."
            )
        self.base_url = base_url
        self.timeout = timeout
        self.max_images = max_images_per_request
        self.quality = image_quality

        print(f"[NVIDIA] Using {self.model_name} at {self.base_url}")

    def _append_image(self, content: list, img: Image.Image, added: int) -> int:
        if added >= self.max_images:
            return added
        b64 = _pil_to_b64(img, self.quality)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
        return added + 1

    def classify(
        self,
        frames_per_view: List[List[Image.Image]],
        prompt: str,
        extra_images: Optional[List[Image.Image]] = None,
    ) -> str:
        content = []
        images_added = 0

        if extra_images:
            content.append(
                {"type": "text", "text": "[Reference examples from training data:]"}
            )
            for img in extra_images:
                images_added = self._append_image(content, img, images_added)

        for v_idx, frames in enumerate(frames_per_view):
            label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            content.append({"type": "text", "text": f"\n[{label}]"})
            for frame in frames:
                images_added = self._append_image(content, frame, images_added)

        content.append({"type": "text", "text": f"\n\n{prompt}"})

        if images_added >= self.max_images:
            print(f"[NVIDIA] Capped images to {self.max_images}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return f"[ERROR] NVIDIA request timed out after {self.timeout}s"
        except requests.exceptions.RequestException as e:
            return f"[ERROR] NVIDIA request failed: {e}"

        data = response.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        return text
