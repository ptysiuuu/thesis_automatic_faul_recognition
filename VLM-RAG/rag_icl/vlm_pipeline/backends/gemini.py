"""
backends/gemini.py
==================
Gemini 3.x Flash backend via Google Generative Language API.

Supported input types per API: text, image, video, audio, PDF.
This backend uses text + images by default. If use_video_mode=True,
it attempts to package frames into a short MP4; if unavailable, it
falls back to images.
"""

import base64
import os
import re
import tempfile
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image

from ..prompts.templates import SYSTEM_PROMPT


def _pil_to_b64(img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_video_mp4(frames: List[Image.Image], fps: int = 2) -> Optional[str]:
    """Try to encode frames into an MP4; return base64 or None on failure."""
    if not frames:
        return None
    try:
        import imageio.v3 as iio
        import numpy as np
    except Exception:
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        frame_arrays = [np.array(f.convert("RGB")) for f in frames]
        iio.imwrite(tmp_path, frame_arrays, fps=fps)
        with open(tmp_path, "rb") as fin:
            data = fin.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


class GeminiBackend:
    """
    Calls Google Generative Language API for Gemini models.

    Parameters
    ----------
    model_name : str
        Gemini model ID, e.g. "gemini-3.5-flash".
    api_key : str
        API key (set GOOGLE_API_KEY or GEMINI_API_KEY).
    base_url : str
        API base URL, default v1beta.
    timeout : int
        Request timeout in seconds.
    max_images_per_request : int
        Cap total images per call to limit payload size.
    image_quality : int
        JPEG quality for base64 encoding.
    use_video_mode : bool
        If True, attempt to send a short MP4 per view instead of images.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash",
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 180,
        max_images_per_request: int = 8,
        image_quality: int = 85,
        use_video_mode: bool = False,
    ):
        self.model_name = model_name
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GOOGLE_API_KEY or GEMINI_API_KEY."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_images = max_images_per_request
        self.quality = image_quality
        self.use_video_mode = use_video_mode

        print(f"[Gemini] Using {self.model_name} at {self.base_url}")

    def _build_parts(
        self,
        frames_per_view: List[List[Image.Image]],
        prompt: str,
        extra_images: Optional[List[Image.Image]] = None,
    ) -> List[dict]:
        parts: List[dict] = []
        images_added = 0

        if extra_images:
            parts.append({"text": "[Reference examples from training data:]"})
            for img in extra_images:
                if images_added >= self.max_images:
                    break
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": _pil_to_b64(img, self.quality),
                        }
                    }
                )
                images_added += 1

        for v_idx, frames in enumerate(frames_per_view):
            label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            parts.append({"text": f"\n[{label}]"})

            if self.use_video_mode:
                video_b64 = _encode_video_mp4(frames, fps=2)
                if video_b64:
                    parts.append(
                        {
                            "inlineData": {
                                "mimeType": "video/mp4",
                                "data": video_b64,
                            }
                        }
                    )
                    continue

            for frame in frames:
                if images_added >= self.max_images:
                    break
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": _pil_to_b64(frame, self.quality),
                        }
                    }
                )
                images_added += 1

        parts.append({"text": f"\n\n{prompt}"})

        if images_added >= self.max_images:
            print(f"[Gemini] Capped images to {self.max_images}")

        return parts

    def classify(
        self,
        frames_per_view: List[List[Image.Image]],
        prompt: str,
        extra_images: Optional[List[Image.Image]] = None,
    ) -> str:
        parts = self._build_parts(frames_per_view, prompt, extra_images)

        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 1.0,
                "maxOutputTokens": 512,
            },
        }

        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return f"[ERROR] Gemini request timed out after {self.timeout}s"
        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                return f"[ERROR] Gemini request failed: {e} | {e.response.text[:500]}"
            return f"[ERROR] Gemini request failed: {e}"

        data = response.json()
        parts_out = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts_out)
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        return text
