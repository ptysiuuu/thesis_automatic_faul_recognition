import argparse
import json
import logging
import os
import signal
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


PROMPT = (
    "Given the following video, describe the actions in the video "
    "and identify the foul contact location. Specify whether the "
    "foul contact region is located in the upper body or lower body."
)

PROCESSOR_FALLBACK = {
    "Video-R1/Video-R1-7B",
    "Video-R1/Qwen2.5-VL-7B-COT-SFT",
}
BASE_PROCESSOR = "Qwen/Qwen2.5-VL-7B-Instruct"


class _Timeout:
    def __init__(self, seconds: Optional[int]):
        self.seconds = seconds
        self._old_handler = None

    def _handler(self, signum, frame):
        raise TimeoutError("Qwen inference timed out")

    def __enter__(self):
        if self.seconds is None or self.seconds <= 0:
            return self
        self._old_handler = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds is None or self.seconds <= 0:
            return False
        signal.alarm(0)
        if self._old_handler is not None:
            signal.signal(signal.SIGALRM, self._old_handler)
        return False


def _load_qwen(model_name: str, quantization: str):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    if quantization in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes is required for quantized Qwen loading. "
                "Install it with `pip install bitsandbytes`."
            ) from exc

        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

    if model_name in PROCESSOR_FALLBACK:
        logging.info(
            "Qwen processor config missing for %s; using %s instead.",
            model_name,
            BASE_PROCESSOR,
        )
        processor_source = BASE_PROCESSOR
    else:
        processor_source = model_name

    processor = AutoProcessor.from_pretrained(
        processor_source, trust_remote_code=True
    )

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )
    except OSError as exc:
        msg = str(exc)
        if "does not appear to have files named" in msg and "safetensors" in msg:
            logging.warning(
                "Qwen safetensors shards missing; retrying with use_safetensors=False"
            )
            model_kwargs_fallback = dict(model_kwargs)
            model_kwargs_fallback["use_safetensors"] = False
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, **model_kwargs_fallback
            )
        else:
            raise
    model.eval()
    return model, processor


def _load_clip(model_name: str, device: str):
    from transformers import CLIPModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def _generate_description(
    model,
    processor,
    mp4_path: str,
    num_frames: int,
    prompt: str,
    max_new_tokens: int,
    timeout_s: Optional[int],
) -> str:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError(
            "qwen_vl_utils is required for Qwen2.5-VL vision inputs."
        ) from exc

    content = [
        {"type": "video", "video": mp4_path, "nframes": num_frames},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    video_kwargs = {
        k: v
        for k, v in (video_kwargs or {}).items()
        if not (k == "fps" and isinstance(v, list) and len(v) == 0)
    }

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to("cuda")

    with _Timeout(timeout_s):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

    generated = out[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return text.strip()


def _encode_text(
    clip_model,
    tokenizer,
    text: str,
    device: str,
) -> np.ndarray:
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1)
    return text_features[0].detach().cpu().numpy().astype(np.float32)


def _iter_actions(data_root: str) -> List[Tuple[str, str]]:
    """Return (action_id, mp4_path) for all actions across Train/Valid/Test."""
    pairs = []
    for split in ("Train", "Valid", "Test"):
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            logging.warning(f"Split directory not found: {split_dir}")
            continue
        for action_id in sorted(os.listdir(split_dir)):
            clip_path = os.path.join(split_dir, action_id, "clip_0.mp4")
            if os.path.exists(clip_path):
                pairs.append((action_id, clip_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Extract Qwen descriptions and CLIP text embeddings from mp4 clips"
    )
    parser.add_argument(
        "--data_root",
        required=True,
        type=str,
        help="Root folder containing Train/Valid/Test subdirectories with mp4 clips",
    )
    parser.add_argument(
        "--output_hdf5",
        required=True,
        type=str,
        help="Output HDF5 path for CLIP text embeddings",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        type=str,
        help="Output JSON path for raw Qwen descriptions",
    )
    parser.add_argument(
        "--qwen_model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        type=str,
    )
    parser.add_argument(
        "--clip_model",
        default="openai/clip-vit-large-patch14",
        type=str,
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "4bit", "8bit"],
        type=str,
    )
    parser.add_argument(
        "--num_frames",
        default=8,
        type=int,
        help="Number of frames to sample from each clip for Qwen",
    )
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--timeout_s", default=60, type=int)
    parser.add_argument(
        "--max_actions",
        default=None,
        type=int,
        help="Stop after this many actions (for testing)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
    )

    logging.info("Loading Qwen and CLIP...")
    qwen, processor = _load_qwen(args.qwen_model, args.quantization)
    clip_model, clip_tokenizer = _load_clip(args.clip_model, device="cuda")

    os.makedirs(os.path.dirname(args.output_hdf5) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    pairs = _iter_actions(args.data_root)
    logging.info(f"Total actions found: {len(pairs)}")

    descriptions: Dict[str, str] = {}
    failures: List[str] = []
    processed_count = 0

    with h5py.File(args.output_hdf5, "a") as out_h5:
        for action_id, mp4_path in tqdm(pairs, desc="Extracting"):
            if args.max_actions is not None and processed_count >= args.max_actions:
                break

            if action_id in out_h5:
                processed_count += 1
                continue

            try:
                text = _generate_description(
                    qwen,
                    processor,
                    mp4_path,
                    args.num_frames,
                    PROMPT,
                    max_new_tokens=args.max_new_tokens,
                    timeout_s=args.timeout_s,
                )
                if not text:
                    raise ValueError("empty description")

                emb = _encode_text(
                    clip_model,
                    clip_tokenizer,
                    text,
                    device="cuda",
                )
                descriptions[action_id] = text

            except Exception as exc:
                logging.warning(f"{action_id}: {exc}")
                failures.append(action_id)
                emb = np.zeros((768,), dtype=np.float32)
                descriptions[action_id] = ""

            out_h5.create_dataset(action_id, data=emb, dtype="float32")
            processed_count += 1

    payload = {
        "prompt": PROMPT,
        "descriptions": descriptions,
        "failures": failures,
    }
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)

    logging.info(
        f"Done. Saved {len(descriptions)} embeddings; {len(failures)} failures."
    )


if __name__ == "__main__":
    main()