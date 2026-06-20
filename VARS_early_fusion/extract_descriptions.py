import argparse
import base64
import json
import logging
import os
import signal
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
read_video = None


PROMPT = (
    "Given the following video clip of a soccer foul, briefly describe "
    "the physical contact between players and state whether the foul "
    "contact region is upper body or lower body. Be concise."
)

THINKING_PROMPT = (
    "Carefully analyze this soccer foul clip. Consider the force of the challenge, "
    "whether there was danger to the opponent, and the exact nature of the physical "
    "contact. Describe the contact between players and state whether the foul "
    "contact region is upper body or lower body."
)

PROCESSOR_FALLBACK = {
    "Video-R1/Video-R1-7B",
    "Video-R1/Qwen2.5-VL-7B-COT-SFT",
}
BASE_PROCESSOR = "Qwen/Qwen2.5-VL-7B-Instruct"

# ---------------------------------------------------------------------------
# VLM registry
# ---------------------------------------------------------------------------

VLM_REGISTRY = {
    "qwen2.5-vl-7b": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "qwen2",
        "gpus": 1,
    },
    "qwen3-vl-8b": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "family": "qwen3",
        "gpus": 1,
    },
    "qwen3-vl-30b": {
        "hf_id": "/net/obelix/homes/aszostek/hf_models/qwen3-vl-30b",
        "family": "qwen3",
        "gpus": 1,
    },
    "qwen3-vl-235b": {
        "hf_id": "Qwen/Qwen3-VL-235B-Instruct",
        "family": "qwen3",
        "gpus": 8,
    },
    "gemma4-12b": {
        "hf_id": "google/gemma-4-12b-it",
        "family": "gemma4",
        "gpus": 1,
    },
    "gemma4-31b": {
        "hf_id": "google/gemma-4-31b-it",
        "family": "gemma4",
        "gpus": 2,
    },
    "qwen3-vl-30b-a3b": {
    "hf_id":  "/net/obelix/homes/aszostek/hf_models/qwen3-vl-30b",
    "family": "qwen3",
    "gpus":   1,
},
}


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
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor ,Qwen3VLMoeForConditionalGeneration

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
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
			model_name,
			torch_dtype=torch.bfloat16,
			device_map="auto",
		)
    except OSError as exc:
        msg = str(exc)
        if "does not appear to have files named" in msg and "safetensors" in msg:
            logging.warning(
                "Qwen safetensors shards missing; retrying with use_safetensors=False"
            )
            model_kwargs_fallback = dict(model_kwargs)
            model_kwargs_fallback["use_safetensors"] = False
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
			model_name,
			torch_dtype=torch.bfloat16,
			device_map="auto",
		)
        else:
            raise
    model.eval()
    return model, processor


def _load_qwen_api():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is required for Qwen API calls. Install it with `pip install openai`."
        ) from exc

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY is not set. Export it before using --use_api."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://ws-ltrh4g9yaar5vco0.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    )
    return client


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> block produced by thinking-mode models."""
    import re

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _load_vlm(model_key: str, quantize: str = "", hf_cache_dir: str = ""):
    """
    Load a VLM from VLM_REGISTRY by short key.

    Returns (model, processor, family) where family is "qwen2", "qwen3", or "gemma4".
    device_map="auto" is always used so SLURM --gres=gpu:N controls GPU count.
    """
    entry = VLM_REGISTRY[model_key]
    hf_id = entry["hf_id"]
    family = entry["family"]

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

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

    if family == "qwen2":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(hf_id, **model_kwargs)
    elif family == "qwen3":
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration

            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(hf_id, **model_kwargs)
        except ImportError:
            logging.warning(
                "Qwen3VLForConditionalGeneration not found in transformers; "
                "falling back to Qwen2_5_VLForConditionalGeneration."
            )
            from transformers import Qwen2_5_VLForConditionalGeneration

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(hf_id, **model_kwargs)
    elif family == "gemma4":
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(hf_id, **model_kwargs)
        except Exception:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
    else:
        raise ValueError(f"Unknown VLM family: {family!r}")

    model.eval()
    return model, processor, family


def _generate_description_gemma(
    model,
    processor,
    mp4_path: str,
    num_frames: int,
    prompt: str,
    max_new_tokens: int,
    timeout_s: Optional[int],
    thinking: bool = False,
) -> str:
    """Generate a description using a Gemma 4 vision-language model."""
    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError("Pillow is required for Gemma inference.") from exc

    frames_np = _sample_frames(mp4_path, num_frames)
    pil_frames = [PILImage.fromarray(frames_np[i]) for i in range(frames_np.shape[0])]

    content = [{"type": "image"} for _ in pil_frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input],
        images=pil_frames,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with _Timeout(timeout_s):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    generated = out[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return text.strip()


def _load_clip(model_name: str, device: str):
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModelWithProjection.from_pretrained(model_name)
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
    family: str = "qwen2",
    thinking: bool = False,
) -> str:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError(
            "qwen_vl_utils is required for Qwen VL vision inputs."
        ) from exc

    # Qwen3 thinking mode: prepend /think to the text content
    effective_prompt = prompt
    if thinking and family == "qwen3":
        effective_prompt = "/think\n" + prompt

    content = [
        {"type": "video", "video": mp4_path, "nframes": num_frames},
        {"type": "text", "text": effective_prompt},
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


def _frames_to_base64(frames: np.ndarray) -> list:
    """Convert [T, H, W, C] uint8 array to list of base64 JPEG data URIs."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for API frame encoding. Install it with `pip install pillow`."
        ) from exc

    result = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i])
        img = img.resize((512, 512))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        result.append(f"data:image/jpeg;base64,{b64}")
    return result


def _generate_description_api(
    client,
    frames: np.ndarray,
    prompt: str,
    max_tokens: int = 256,
    model_name: str = "qwen3-vl-235b-a22b-thinking",
) -> str:
    data_uris = _frames_to_base64(frames)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": data_uris},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_tokens,
    )

    text = response.choices[0].message.content
    # Strip the thinking block from Qwen thinking models if present.
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _sample_frames(mp4_path: str, num_frames: int) -> np.ndarray:
    video, _, _ = read_video(mp4_path, pts_unit="sec", output_format="THWC")
    if video.shape[0] == 0:
        raise ValueError("empty video")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")

    total = video.shape[0]
    if total == num_frames:
        return video.numpy()

    if total < num_frames:
        pad = video[-1:].repeat(num_frames - total, 1, 1, 1)
        video = torch.cat([video, pad], dim=0)
        return video.numpy()

    indices = np.linspace(0, total - 1, num_frames)
    indices = np.round(indices).astype(int)
    return video[indices].numpy()


def _encode_text(clip_model, clip_tokenizer, text: str, device: str) -> np.ndarray:
    inputs = clip_tokenizer(
        text, padding=True, truncation=True,
        max_length=77, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
        text_features = outputs.text_embeds
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
                key = action_id if split == "Train" else f"{split}_{action_id}"
                pairs.append((key, clip_path))
    return pairs


def _iter_actions_hdf5(hdf5_dir: str) -> List[Tuple[str, str, str]]:
    """
    Return (output_key, hdf5_path, action_id) for all actions across splits.

    output_key  — the key written to the output text-embedding HDF5.
    hdf5_path   — path to the compact HDF5 that contains the frames.
    action_id   — group name inside that HDF5 (e.g. 'action_0').
    """
    pairs = []
    for split in ("Train", "Valid", "Test"):
        hdf5_path = os.path.join(hdf5_dir, f"{split}.hdf5")
        if not os.path.exists(hdf5_path):
            logging.warning(f"HDF5 not found, skipping: {hdf5_path}")
            continue
        with h5py.File(hdf5_path, "r") as f:
            action_ids = sorted(f.keys())
        for action_id in action_ids:
            key = action_id if split == "Train" else f"{split}_{action_id}"
            pairs.append((key, hdf5_path, action_id))
    return pairs


def _load_pil_frames_from_hdf5(
    hdf5_path: str,
    action_id: str,
    clip_key: str = "clip_0",
    num_frames: int = 8,
) -> Optional[List]:
    """Load evenly-spaced PIL frames from one clip in a compact HDF5."""
    try:
        from PIL import Image as _PILImage
    except ImportError:
        return None

    with h5py.File(hdf5_path, "r") as f:
        key = f"{action_id}/{clip_key}"
        if key not in f:
            return None
        frames_np = f[key][:]  # [T, H, W, C] uint8

    T = frames_np.shape[0]
    if T == 0:
        return None

    indices = np.linspace(0, T - 1, min(num_frames, T), dtype=int)
    return [_PILImage.fromarray(frames_np[i]) for i in indices]


def _generate_description_from_frames(
    model,
    processor,
    pil_frames: list,
    prompt: str,
    max_new_tokens: int,
    timeout_s: Optional[int],
    family: str = "qwen2",
    thinking: bool = False,
) -> str:
    """Generate a description using pre-extracted PIL frames (no mp4 needed)."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError("qwen_vl_utils is required for Qwen VL inference.") from exc

    effective_prompt = prompt
    if thinking and family == "qwen3":
        effective_prompt = "/think\n" + prompt

    content = [{"type": "image", "image": f} for f in pil_frames]
    content.append({"type": "text", "text": effective_prompt})

    messages = [{"role": "user", "content": content}]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract VLM descriptions and CLIP text embeddings from mp4 clips"
    )

    # --- VLM selection (new registry-based API) ---
    parser.add_argument(
        "--vlm_model",
        default=None,
        choices=list(VLM_REGISTRY.keys()),
        type=str,
        help=(
            "VLM to use for description generation. "
            "Choices: " + ", ".join(VLM_REGISTRY.keys())
        ),
    )
    parser.add_argument(
        "--vlm_quantize",
        default=None,
        choices=["4bit", "8bit"],
        type=str,
        help="Optional quantization for local VLM (4bit or 8bit)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "Enable thinking/chain-of-thought mode. "
            "For Qwen3 models adds /think prefix; strips <think> blocks before CLIP encoding."
        ),
    )

    # --- Path args ---
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="SoccerNet_Data root with Train/Valid/Test mp4 clips. "
             "Required when NOT using --hdf5_dir.",
    )
    parser.add_argument(
        "--hdf5_dir",
        default=None,
        type=str,
        help="Directory with pre-extracted compact HDF5 files (Train.hdf5 / Valid.hdf5 / "
             "Test.hdf5). When provided, frames are read from HDF5 instead of mp4 files "
             "(e.g. SoccerNet_HDF5_compact). Supersedes --data_dir for frame reading.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help=(
            "Output directory. HDF5 and JSON filenames are derived automatically: "
            "text_embeddings_{vlm_model}.h5 / descriptions_{vlm_model}.json"
        ),
    )
    parser.add_argument(
        "--hf_cache_dir",
        default="",
        type=str,
        help="Override HF_HOME and TRANSFORMERS_CACHE to this directory",
    )

    # --- Legacy / backward-compat args ---
    parser.add_argument(
        "--data_root",
        default=None,
        type=str,
        help="(Legacy) Root folder with Train/Valid/Test mp4 clips. Use --data_dir instead.",
    )
    parser.add_argument(
        "--output_hdf5",
        default=None,
        type=str,
        help="(Legacy) Explicit output HDF5 path. Overrides --output_dir derived name.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        type=str,
        help="(Legacy) Explicit output JSON path. Overrides --output_dir derived name.",
    )
    parser.add_argument(
        "--qwen_model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        type=str,
        help="(Legacy) HuggingFace model ID for Qwen. Use --vlm_model instead.",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "4bit", "8bit"],
        type=str,
        help="(Legacy) Quantization for --qwen_model. Use --vlm_quantize instead.",
    )
    parser.add_argument(
        "--use_api",
        action="store_true",
        help="Use Qwen API instead of local model weights",
    )
    parser.add_argument(
        "--api_model",
        default="qwen3-vl-235b-a22b-thinking",
        type=str,
        help="Qwen API model name (used with --use_api)",
    )
    parser.add_argument(
        "--clip_model",
        default="openai/clip-vit-large-patch14",
        type=str,
    )
    parser.add_argument(
        "--num_frames",
        default=8,
        type=int,
        help="Number of frames to sample from each clip",
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

    # Set HF cache dirs before any HF imports
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir

    # Resolve frame source: HDF5 compact takes priority over raw mp4
    use_hdf5 = bool(args.hdf5_dir)
    if use_hdf5:
        hdf5_root = os.path.abspath(os.path.expanduser(os.path.expandvars(args.hdf5_dir)))
        data_root = None
    else:
        raw_data_root = args.data_dir or args.data_root
        if not raw_data_root:
            parser.error("Provide --hdf5_dir (compact HDF5) or --data_dir (mp4 root).")
        data_root = os.path.abspath(os.path.expanduser(os.path.expandvars(raw_data_root)))
        hdf5_root = None

    # Resolve output paths
    model_suffix = args.vlm_model if args.vlm_model else "custom"
    if args.output_hdf5:
        output_hdf5 = os.path.abspath(os.path.expanduser(os.path.expandvars(args.output_hdf5)))
    elif args.output_dir:
        output_hdf5 = os.path.join(args.output_dir, f"text_embeddings_{model_suffix}.h5")
    else:
        parser.error("Provide --output_dir or (legacy) --output_hdf5.")

    if args.output_json:
        output_json = os.path.abspath(os.path.expanduser(os.path.expandvars(args.output_json)))
    elif args.output_dir:
        output_json = os.path.join(args.output_dir, f"descriptions_{model_suffix}.json")
    else:
        parser.error("Provide --output_dir or (legacy) --output_json.")

    os.makedirs(os.path.dirname(os.path.abspath(output_hdf5)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)

    # Select prompt
    active_prompt = THINKING_PROMPT if args.thinking else PROMPT

    # Load models
    logging.info("Loading VLM and CLIP...")
    vlm = None
    processor = None
    vlm_family = "qwen2"
    client = None

    if args.use_api:
        client = _load_qwen_api()
    elif args.vlm_model:
        quantize = args.vlm_quantize or ""
        vlm, processor, vlm_family = _load_vlm(args.vlm_model, quantize=quantize)
    else:
        quantize = args.quantization if args.quantization != "none" else ""
        vlm, processor = _load_qwen(args.qwen_model, quantize)
        vlm_family = "qwen2"

    clip_model, clip_tokenizer = _load_clip(args.clip_model, device="cuda")

    if use_hdf5:
        hdf5_pairs = _iter_actions_hdf5(hdf5_root)
        logging.info(f"Total actions found in HDF5: {len(hdf5_pairs)}")
        pairs = None
    else:
        pairs = _iter_actions(data_root)
        logging.info(f"Total actions found: {len(pairs)}")
        hdf5_pairs = None

    descriptions: Dict[str, str] = {}
    failures: List[str] = []
    processed_count = 0

    iter_source = hdf5_pairs if use_hdf5 else pairs

    with h5py.File(output_hdf5, "a") as out_h5:
        for item in tqdm(iter_source, desc="Extracting"):
            if args.max_actions is not None and processed_count >= args.max_actions:
                break

            # Unpack depending on mode
            if use_hdf5:
                action_id, src_hdf5_path, raw_action_id = item
            else:
                action_id, mp4_path = item

            if action_id in out_h5:
                processed_count += 1
                continue

            try:
                if use_hdf5:
                    # Read pre-extracted frames from compact HDF5
                    pil_frames = _load_pil_frames_from_hdf5(
                        src_hdf5_path, raw_action_id, "clip_0", args.num_frames
                    )
                    if not pil_frames:
                        raise ValueError("clip_0 not found in HDF5")
                    if args.use_api:
                        import numpy as _np
                        frames_np = _np.stack([_np.array(f) for f in pil_frames])
                        text = _generate_description_api(
                            client, frames_np, active_prompt,
                            max_tokens=args.max_new_tokens, model_name=args.api_model,
                        )
                    elif vlm_family == "gemma4":
                        # Gemma uses a different processor; HDF5 mode not yet supported
                        raise NotImplementedError(
                            "gemma4 + --hdf5_dir is not supported; use --data_dir with mp4s."
                        )
                    else:
                        # qwen2 / qwen3 — process_vision_info handles PIL image content
                        text = _generate_description_from_frames(
                            vlm, processor, pil_frames, active_prompt,
                            max_new_tokens=args.max_new_tokens,
                            timeout_s=args.timeout_s, family=vlm_family,
                            thinking=args.thinking,
                        )
                else:
                    # Original mp4 path
                    if args.use_api:
                        sampled_frames = _sample_frames(mp4_path, args.num_frames)
                        text = _generate_description_api(
                            client, sampled_frames, active_prompt,
                            max_tokens=args.max_new_tokens, model_name=args.api_model,
                        )
                    elif vlm_family == "gemma4":
                        text = _generate_description_gemma(
                            vlm, processor, mp4_path, args.num_frames, active_prompt,
                            max_new_tokens=args.max_new_tokens,
                            timeout_s=args.timeout_s, thinking=args.thinking,
                        )
                    else:
                        text = _generate_description(
                            vlm, processor, mp4_path, args.num_frames, active_prompt,
                            max_new_tokens=args.max_new_tokens,
                            timeout_s=args.timeout_s, family=vlm_family,
                            thinking=args.thinking,
                        )

                if not text:
                    raise ValueError("empty description")

                # Strip thinking block before CLIP encoding
                text_for_clip = _strip_thinking(text) if args.thinking else text
                if not text_for_clip:
                    text_for_clip = text

                emb = _encode_text(clip_model, clip_tokenizer, text_for_clip, device="cuda")
                descriptions[action_id] = text

            except Exception as exc:
                logging.warning(f"{action_id}: {exc}")
                failures.append(action_id)
                emb = np.zeros((768,), dtype=np.float32)
                descriptions[action_id] = ""

            out_h5.create_dataset(action_id, data=emb, dtype="float32")
            processed_count += 1

    payload = {
        "prompt": active_prompt,
        "vlm_model": args.vlm_model or args.qwen_model,
        "thinking": args.thinking,
        "descriptions": descriptions,
        "failures": failures,
    }
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    logging.info(
        f"Done. Saved {len(descriptions)} embeddings to {output_hdf5}; "
        f"{len(failures)} failures."
    )


if __name__ == "__main__":
    main()