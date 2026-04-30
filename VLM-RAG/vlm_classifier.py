"""
vlm_classifier.py
=================
VLM-based foul classifier supporting 4 prompting strategies:
  A) zero_shot        — direct classification, no rules
  B) rule_grounded    — FIFA Law 12 RAG context injected
  C) chain_of_thought — step-by-step reasoning before answer
  D) few_shot         — in-context examples + rules

Supports:
  - Qwen2.5-VL-7B-Instruct  (local, recommended)
  - InternVL2.5-8B           (local, alternative)
  - GPT-4o                   (API, via openai)
  - Gemini-2.0-flash         (API, via google-generativeai)

Dependencies:
  pip install transformers torch torchvision pillow numpy
  pip install qwen-vl-utils          # for Qwen
  pip install openai                 # for GPT-4o (optional)
  pip install google-generativeai    # for Gemini (optional)
"""

import os
import re
import json
import base64
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from io import BytesIO

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Label mappings (must match your dataset)
# ---------------------------------------------------------------------------

ACTION_CLASSES = [
    "Tackling", "Standing tackling", "High leg", "Holding",
    "Pushing", "Elbowing", "Challenge", "Dive",
]
SEVERITY_CLASSES = ["No offence", "No card", "Yellow card", "Red card"]

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}
SEVERITY_TO_IDX = {s: i for i, s in enumerate(SEVERITY_CLASSES)}


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_keyframes(
    hdf5_file,
    action_key: str,
    clip_key: str,
    n_frames: int = 4,
    start_frame: int = 63,
    end_frame: int = 87,
) -> List[Image.Image]:
    """
    Extract N evenly-spaced keyframes from HDF5 clip.

    Parameters
    ----------
    hdf5_file : h5py.File (already open)
    action_key : str  e.g. "action_0"
    clip_key   : str  e.g. "clip_0"
    n_frames   : int  keyframes to extract per clip
    """
    import h5py
    key = f"{action_key}/{clip_key}"
    if key not in hdf5_file:
        return []

    frames_np = hdf5_file[key][:]  # [T, H, W, C] uint8

    # Sample evenly
    total = len(frames_np)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames_pil = []
    for idx in indices:
        img = Image.fromarray(frames_np[idx])
        frames_pil.append(img)
    return frames_pil


def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ACTION_LIST_STR = "\n".join(f"  - {a}" for a in ACTION_CLASSES)
SEVERITY_LIST_STR = "\n".join(f"  - {s}" for s in SEVERITY_CLASSES)

SYSTEM_PROMPT = """You are an expert football referee assistant.
Your task is to analyze video frames from multiple camera angles
and classify football foul incidents according to FIFA Laws of the Game.
Always respond with ONLY a JSON object — no other text."""

ZERO_SHOT_TEMPLATE = """You are analyzing a potential football foul from {n_views} camera angles.
The frames below show the incident from different perspectives.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

Classify the incident:

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""

RULE_GROUNDED_TEMPLATE = """You are analyzing a potential football foul from {n_views} camera angles.
The frames below show the incident from different perspectives.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

Based on the FIFA rules above, classify the incident:

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Apply the rules strictly:
- EXCESSIVE FORCE or endangering safety → RED CARD
- RECKLESS (disregard for opponent) → YELLOW CARD  
- CARELESS (lack of attention) → No card but still a foul
- SIMULATION/DIVING → No offence, Yellow card

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence citing the applicable rule>"}}"""

COT_TEMPLATE = """You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

Think step by step, then classify. Your steps:
1. What body part made contact (or was there any contact)?
2. Was the challenge from front, side, or behind?
3. Was the player attempting to play the ball?
4. How much force was used — careless, reckless, or excessive?
5. Was this simulation/diving?
6. Based on steps 1-5, what is the action type?
7. Based on steps 1-5, what is the severity?

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Respond with ONLY this JSON:
{{"step1": "...", "step2": "...", "step3": "...", "step4": "...",
  "step5": "...", "step6": "...", "step7": "...",
  "action": "<action type>", "severity": "<severity>"}}"""

FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Incident: Player lunges from behind with foot raised high, making full contact with opponent's leg.
Classification: {{"action": "Tackling", "severity": "Red card",
  "reasoning": "Tackle from behind with excessive force endangers opponent's safety — serious foul play."}}

EXAMPLE 2:
Incident: Player extends elbow into opponent's face while not challenging for the ball.
Classification: {{"action": "Elbowing", "severity": "Red card",
  "reasoning": "Elbowing is violent conduct regardless of ball proximity."}}

EXAMPLE 3:
Incident: Player falls dramatically after minimal contact, exaggerating the effect.
Classification: {{"action": "Dive", "severity": "No offence",
  "reasoning": "Simulation — player feigned injury to deceive referee, yellow card for unsporting behaviour."}}

EXAMPLE 4:
Incident: Player grabs opponent's shirt to slow them down during a counterattack.
Classification: {{"action": "Holding", "severity": "Yellow card",
  "reasoning": "Reckless holding — disregards opponent's progress, caution warranted."}}

"""

FEW_SHOT_TEMPLATE = """You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

Here are some examples of correctly classified incidents:
{examples}

Now classify the incident shown in the video frames:

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""


def build_prompt(
    strategy: str,
    n_views: int,
    law12_context: str = "",
) -> str:
    kwargs = dict(
        n_views=n_views,
        action_list=ACTION_LIST_STR,
        severity_list=SEVERITY_LIST_STR,
        law12_context=law12_context,
    )
    if strategy == "zero_shot":
        return ZERO_SHOT_TEMPLATE.format(**kwargs)
    elif strategy == "rule_grounded":
        return RULE_GROUNDED_TEMPLATE.format(**kwargs)
    elif strategy == "chain_of_thought":
        return COT_TEMPLATE.format(**kwargs)
    elif strategy == "few_shot":
        return FEW_SHOT_TEMPLATE.format(examples=FEW_SHOT_EXAMPLES, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_response(response_text: str) -> Tuple[int, int]:
    """
    Parse VLM JSON response into (action_idx, severity_idx).
    Returns (-1, -1) if parsing fails.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```json\s*|\s*```", "", response_text).strip()

    # Find JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return -1, -1

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        # Try to fix common issues
        try:
            fixed = match.group().replace("'", '"')
            data = json.loads(fixed)
        except Exception:
            return -1, -1

    action_str   = data.get("action", "")
    severity_str = data.get("severity", "")

    # Fuzzy match action
    action_idx = -1
    for i, a in enumerate(ACTION_CLASSES):
        if a.lower() in action_str.lower() or action_str.lower() in a.lower():
            action_idx = i
            break

    # Fuzzy match severity
    severity_idx = -1
    for i, s in enumerate(SEVERITY_CLASSES):
        if s.lower() in severity_str.lower() or severity_str.lower() in s.lower():
            severity_idx = i
            break

    return action_idx, severity_idx


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

class QwenVLBackend:
    """
    Qwen2.5-VL-7B-Instruct backend.
    Accepts multiple images per prompt — ideal for multi-view.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"[QwenVL] Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        self.model.eval()
        print("[QwenVL] Ready.")

    def classify(
        self,
        frames_per_view: List[List[Image.Image]],
        prompt: str,
    ) -> str:
        """
        frames_per_view : list of lists — [[view0_frames], [view1_frames], ...]
        """
        from qwen_vl_utils import process_vision_info

        # Build content: interleave view label + frames
        content = []
        for v_idx, frames in enumerate(frames_per_view):
            view_label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            content.append({"type": "text", "text": f"\n[{view_label}]"})
            for frame in frames:
                content.append({"type": "image", "image": frame})

        content.append({"type": "text", "text": f"\n\n{prompt}"})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only new tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0]


class InternVLBackend:
    """
    InternVL2.5-8B backend.
    Alternative to Qwen, similar multi-image support.
    """

    def __init__(self, model_name: str = "OpenGVLab/InternVL2_5-8B"):
        from transformers import AutoModel, AutoTokenizer
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        print(f"[InternVL] Loading {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.transform = self._build_transform()
        print("[InternVL] Ready.")

    def _build_transform(self, input_size=448):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((input_size, input_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def classify(
        self,
        frames_per_view: List[List[Image.Image]],
        prompt: str,
    ) -> str:
        # Flatten all frames, build <image> tags
        all_frames = []
        image_tags = ""
        for v_idx, frames in enumerate(frames_per_view):
            view_label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            image_tags += f"\n[{view_label}] "
            for frame in frames:
                all_frames.append(frame)
                image_tags += "<image> "

        pixel_values = torch.stack([
            self.transform(f) for f in all_frames
        ]).to(torch.bfloat16).cuda()

        full_prompt = f"{SYSTEM_PROMPT}\n\n{image_tags}\n\n{prompt}"
        generation_config = dict(max_new_tokens=512, do_sample=False)

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            full_prompt,
            generation_config,
        )
        return response


class GPT4oBackend:
    """
    GPT-4o API backend.
    Requires: pip install openai
    Set OPENAI_API_KEY environment variable.
    """

    def __init__(self, model: str = "gpt-4o"):
        import openai
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        print(f"[GPT4o] Using model: {model}")

    def classify(
        self,
        frames_per_view: List[List[Image.Image]],
        prompt: str,
    ) -> str:
        content = [{"type": "text", "text": SYSTEM_PROMPT + "\n\n"}]

        for v_idx, frames in enumerate(frames_per_view):
            view_label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            content.append({"type": "text", "text": f"\n[{view_label}]"})
            for frame in frames:
                b64 = pil_to_base64(frame, fmt="JPEG")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                  "detail": "high"},
                })

        content.append({"type": "text", "text": f"\n\n{prompt}"})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class VLMFoulClassifier:
    """
    Main interface. Wraps backend + RAG + prompt construction + evaluation.

    Parameters
    ----------
    backend : str
        One of: "qwen", "internvl", "gpt4o"
    strategy : str
        One of: "zero_shot", "rule_grounded", "chain_of_thought", "few_shot"
    law12_pdf : str or None
        Path to FIFA Laws PDF. If None, uses hardcoded passages.
    frames_per_view : int
        Number of keyframes to extract per view (default 4).
    """

    def __init__(
        self,
        backend: str = "qwen",
        strategy: str = "rule_grounded",
        law12_pdf: str = None,
        frames_per_view: int = 4,
    ):
        self.strategy = strategy
        self.frames_per_view = frames_per_view

        # Initialize RAG
        from law12_rag import Law12RAG
        use_emb = (backend != "gpt4o")  # skip heavy embedding model for API backends
        self.rag = Law12RAG(
            pdf_path=law12_pdf,
            top_k=3,
            use_embeddings=use_emb,
        )

        # Initialize backend
        if backend == "qwen":
            self.backend = QwenVLBackend()
        elif backend == "internvl":
            self.backend = InternVLBackend()
        elif backend == "gpt4o":
            self.backend = GPT4oBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def classify_action(
        self,
        frames_per_view: List[List[Image.Image]],
        action_hint: str = "Dont know",
    ) -> Tuple[int, int, str]:
        """
        Classify a single multi-view action.

        Parameters
        ----------
        frames_per_view : List[List[Image.Image]]
            One list per view, each containing PIL frames.
        action_hint : str
            Rough action type hint for RAG retrieval (can be "Dont know").

        Returns
        -------
        (action_idx, severity_idx, raw_response)
        """
        # Build RAG context
        if self.strategy != "zero_shot":
            query = self.rag.build_query(action_hint)
            law12_context = self.rag.retrieve(query)
        else:
            law12_context = ""

        prompt = build_prompt(
            strategy=self.strategy,
            n_views=len(frames_per_view),
            law12_context=law12_context,
        )

        raw = self.backend.classify(frames_per_view, prompt)
        action_idx, severity_idx = parse_response(raw)
        return action_idx, severity_idx, raw
