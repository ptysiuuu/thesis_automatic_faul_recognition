import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Optional
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from ..prompts.templates import SYSTEM_PROMPT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def dynamic_preprocess(image, min_num=1, max_num=2, image_size=448):
    """
    Stripped-down tiling — max_num=2 means at most 2 tiles per image.
    With 4 views × 4 frames = 16 images × 2 tiles = 32 tiles total.
    At ~0.5GB per tile = ~16GB for images + 16GB model = ~32GB, fits on 40GB.
    """
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h

    # Only tile if strongly landscape or portrait, otherwise use single tile
    if 0.5 < aspect < 2.0 or max_num == 1:
        return [image.resize((image_size, image_size))]

    # Two tiles: split along dominant axis
    if aspect >= 2.0:
        half = orig_w // 2
        tiles = [image.crop((0, 0, half, orig_h)),
                 image.crop((half, 0, orig_w, orig_h))]
    else:
        half = orig_h // 2
        tiles = [image.crop((0, 0, orig_w, half)),
                 image.crop((0, half, orig_w, orig_h))]

    return [t.resize((image_size, image_size)) for t in tiles]


class InternVLBackend:
    MAX_TILES_PER_IMAGE = 1   # 1 tile = single resize, safest for memory
    IMAGE_SIZE = 448

    def __init__(self, model_name: str = "OpenGVLab/InternVL3-8B"):
        print(f"[InternVL3] Loading {model_name} (max_tiles={self.MAX_TILES_PER_IMAGE})...")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,          # not installed in phi4 env
            trust_remote_code=True,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False)
        self.transform = build_transform(input_size=self.IMAGE_SIZE)
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"[InternVL3] Ready. Model VRAM: {vram:.2f} GB")

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        tiles = dynamic_preprocess(
            image, max_num=self.MAX_TILES_PER_IMAGE,
            image_size=self.IMAGE_SIZE)
        return torch.stack([self.transform(t) for t in tiles])

    def classify(self, frames_per_view: List[List[Image.Image]],
                 prompt: str,
                 extra_images: Optional[List[Image.Image]] = None) -> str:
        pixel_values_list = []
        num_patches_list  = []
        text_prefix = ""

        if extra_images:
            text_prefix += "[Reference examples from training data:]\n"
            for img in extra_images:
                pv = self._process_image(img)
                pixel_values_list.append(pv)
                num_patches_list.append(pv.shape[0])
                text_prefix += "<image>\n"

        for v_idx, frames in enumerate(frames_per_view):
            label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            text_prefix += f"\n[{label}]\n"
            for frame in frames:
                pv = self._process_image(frame)
                pixel_values_list.append(pv)
                num_patches_list.append(pv.shape[0])
                text_prefix += "<image>\n"

        pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).cuda()
        question = f"{SYSTEM_PROMPT}\n\n{text_prefix}\n\n{prompt}"
        gen_config = dict(max_new_tokens=512, do_sample=False)

        response, _ = self.model.chat(
            self.tokenizer, pixel_values, question, gen_config,
            num_patches_list=num_patches_list,
            history=None, return_history=True,
        )
        return response
