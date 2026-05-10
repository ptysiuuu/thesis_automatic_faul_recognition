"""
backends/ollama.py
==================
Drop-in replacement for QwenVLBackend that calls a locally-hosted
Ollama instance instead of loading model weights into GPU memory.

Setup on your PC:
    ollama serve
    ollama pull qwen2.5vl:7b      # or whatever tag you pulled

Default endpoint: http://localhost:11434
Override with OLLAMA_HOST env var or --ollama_host CLI arg.

The classify() signature is identical to QwenVLBackend so all
existing strategies (cos_two_stage, static_few_shot, etc.) work
without any changes.

Images are sent as base64-encoded JPEGs in the Ollama /api/chat
multimodal format.
"""

import os
import base64
import json
import re
import time
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image

from ..prompts.templates import SYSTEM_PROMPT


def _pil_to_b64(img: Image.Image, quality: int = 85) -> str:
    """Convert PIL image to base64 JPEG string for Ollama."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class OllamaBackend:
    """
    Calls Ollama's /api/chat endpoint with multimodal messages.

    Parameters
    ----------
    model_name : str
        Ollama model tag, e.g. "qwen2.5vl:7b" or "qwen2.5vl:7b-instruct-q4_K_M"
        Run `ollama list` to see available models.
    host : str
        Base URL of the Ollama server (default: http://localhost:11434)
    timeout : int
        Request timeout in seconds (default: 120 — vision inference is slow)
    max_images_per_request : int
        Cap total images per API call to avoid OOM on CPU/consumer GPU.
        Extra images are dropped (keeping reference examples, then views in order).
        Default 16 = 4 views × 4 frames, same as cluster setup.
    image_quality : int
        JPEG quality for base64 encoding (lower = faster transfer, less detail).
        85 is a good balance; use 70 if you notice slowness.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5vl:7b",
        host: str = None,
        timeout: int = 180,
        max_images_per_request: int = 16,
        image_quality: int = 85,
    ):
        self.model_name = model_name
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.max_images = max_images_per_request
        self.quality = image_quality
        self.endpoint = f"{self.host}/api/chat"

        print(f"[Ollama] Connecting to {self.host} — model: {self.model_name}")
        self._check_connection()

    def _check_connection(self):
        """Verify Ollama is reachable and model is available."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model_name in m for m in models):
                print(f"[Ollama] WARNING: '{self.model_name}' not found in {models}")
                print(f"[Ollama] Run: ollama pull {self.model_name}")
            else:
                print(f"[Ollama] Model '{self.model_name}' is available.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"[Ollama] Cannot connect to {self.host}. "
                "Is Ollama running? Start with: ollama serve"
            )

    def classify(
        self,
        frames_per_view: List[List[Image.Image]],
        prompt: str,
        extra_images: Optional[List[Image.Image]] = None,
    ) -> str:
        """
        Same interface as QwenVLBackend.classify().

        Builds a single user message with:
          - reference example images (extra_images, if any)
          - per-view frame images with [Live camera] / [Replay N] labels
          - the prompt text

        All images go into the 'images' list of the user message.
        The text contains [View labels] to help the model understand layout.
        """
        all_images_b64: List[str] = []
        text_parts: List[str] = []

        # Reference examples (medoid cache images)
        if extra_images:
            text_parts.append("[Reference examples from training data:]")
            for img in extra_images:
                all_images_b64.append(_pil_to_b64(img, self.quality))

        # Multi-view frames
        for v_idx, frames in enumerate(frames_per_view):
            label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            text_parts.append(f"\n[{label}]")
            for frame in frames:
                all_images_b64.append(_pil_to_b64(frame, self.quality))

        text_parts.append(f"\n\n{prompt}")
        full_text = "\n".join(text_parts)

        # Cap images to avoid memory issues
        if len(all_images_b64) > self.max_images:
            print(
                f"[Ollama] Capping {len(all_images_b64)} images to {self.max_images}"
            )
            all_images_b64 = all_images_b64[: self.max_images]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": full_text,
                "images": all_images_b64,
            },
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 512,
            },
        }

        t0 = time.time()
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return f"[ERROR] Ollama request timed out after {self.timeout}s"
        except requests.exceptions.RequestException as e:
            return f"[ERROR] Ollama request failed: {e}"

        elapsed = time.time() - t0
        data = response.json()
        text = data.get("message", {}).get("content", "")

        # Strip markdown code fences if Ollama wraps JSON
        text = re.sub(r"```json\s*|\s*```", "", text).strip()

        print(f"[Ollama] {elapsed:.1f}s | images={len(all_images_b64)} | {len(text)} chars")
        return text
