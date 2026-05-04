
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Optional
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from ..prompts.templates import SYSTEM_PROMPT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

class InternVLBackend:
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-14B"):
        print(f"[InternVL3] Loading {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.transform = build_transform(input_size=448)
        print("[InternVL3] Ready.")

    def _process_pil_image(self, image: Image.Image, max_num=6) -> torch.Tensor:
        # Reduced max_num to 6 to save tokens since we process many frames
        images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=max_num)
        pixel_values = [self.transform(img) for img in images]
        return torch.stack(pixel_values)

    def classify(self, frames_per_view: List[List[Image.Image]], prompt: str, extra_images: Optional[List[Image.Image]] = None) -> str:
        pixel_values_list = []
        num_patches_list = []
        
        # 1. Process Medoid Examples (if any)
        text_prefix = ""
        if extra_images:
            text_prefix += "[Reference examples from training data:]\n"
            for img in extra_images:
                pv = self._process_pil_image(img, max_num=2) # Keep examples small
                pixel_values_list.append(pv)
                num_patches_list.append(pv.shape[0])
                text_prefix += "<image>\n"
        
        # 2. Process Live Frames
        for v_idx, frames in enumerate(frames_per_view):
            label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            text_prefix += f"\n[{label}]\n"
            for frame in frames:
                pv = self._process_pil_image(frame, max_num=6)
                pixel_values_list.append(pv)
                num_patches_list.append(pv.shape[0])
                text_prefix += "<image>\n"

        # Combine all pixel tensors
        pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).cuda()
        
        # Construct the final question
        question = f"{SYSTEM_PROMPT}\n\n{text_prefix}\n\n{prompt}"
        
        generation_config = dict(max_new_tokens=512, do_sample=False)
        
        # InternVL3 requires num_patches_list when passing multiple independent images
        response, _ = self.model.chat(
            self.tokenizer, 
            pixel_values, 
            question, 
            generation_config,
            num_patches_list=num_patches_list,
            history=None, 
            return_history=True
        )
        
        return response
