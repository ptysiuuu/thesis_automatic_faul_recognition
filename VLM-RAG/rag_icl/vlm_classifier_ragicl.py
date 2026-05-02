"""
vlm_classifier_ragicl.py
========================
VLM foul classifier with RAG-ICL (Retrieval-Augmented In-Context Learning).

Four strategies:
  zero_shot      — no context, direct classification
  rule_grounded  — FIFA Law 12 RAG context only
  static_few_shot— fixed handcrafted examples (baseline)
  rag_icl        — dynamic examples retrieved by MViT visual similarity (novel)

The rag_icl strategy:
  1. Extracts MViT-v2-S features from the test clip (live view)
  2. Searches the FAISS index built from Train split
  3. Retrieves K most visually similar historical fouls with their labels
  4. Injects these as in-context examples alongside FIFA Law 12 rules
  5. Feeds everything to Qwen2.5-VL-7B for final classification

This is the novel contribution — combining visual retrieval with VLM prompting.
"""

import os
import re
import json
import base64
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

ACTION_CLASSES = [
    "Tackling",
    "Standing tackling",
    "High leg",
    "Holding",
    "Pushing",
    "Elbowing",
    "Challenge",
    "Dive",
]
SEVERITY_CLASSES = ["No offence", "No card", "Yellow card", "Red card"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}
SEVERITY_TO_IDX = {s: i for i, s in enumerate(SEVERITY_CLASSES)}


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------


def extract_keyframes(
    hdf5_file, action_key: str, clip_key: str, n_frames: int = 4
) -> List[Image.Image]:
    key = f"{action_key}/{clip_key}"
    if key not in hdf5_file:
        return []
    frames_np = hdf5_file[key][:]
    total = len(frames_np)
    if total < 2:
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    return [Image.fromarray(frames_np[i]) for i in indices]


def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Law 12 hardcoded fallback
# ---------------------------------------------------------------------------

LAW12_HARDCODED = """
=== FIFA Law 12: Fouls and Misconduct ===

CARELESS: lack of attention when making a challenge. No disciplinary sanction.
RECKLESS: disregard to the danger to an opponent. YELLOW CARD.
USING EXCESSIVE FORCE: far exceeded necessary force, endangered opponent. RED CARD.

DIRECT FREE KICK offences: tackling, kicking, jumping at, charging, striking,
pushing, holding, impeding an opponent.

RED CARD — SENDING OFF:
- Serious foul play: tackle using excessive force or brutality
- Violent conduct: excessive force against anyone not challenging for ball
- Denying obvious goal-scoring opportunity (DOGSO)

YELLOW CARD — CAUTION:
- Reckless challenges
- Unsporting behaviour (including simulation/diving)

HIGH LEG / HIGH FOOT:
Raising foot dangerously near opponent's head → if excessive force: RED CARD.

ELBOWING: violent conduct regardless of ball proximity → RED CARD.

DIVING / SIMULATION: feigning injury or being fouled → YELLOW CARD.

TACKLING FROM BEHIND endangering safety → RED CARD.
RECKLESS TACKLE → YELLOW CARD.
CARELESS TACKLE → free kick, no card.
"""


# ---------------------------------------------------------------------------
# Law 12 RAG
# ---------------------------------------------------------------------------


class Law12RAG:
    def __init__(
        self, pdf_path: str = None, top_k: int = 3, use_embeddings: bool = True
    ):
        self.top_k = top_k
        self.use_embeddings = use_embeddings
        self.chunks = []
        self.embeddings = None
        self._model = None

        text = self._load_text(pdf_path)
        self.chunks = self._chunk(text, chunk_size=400)
        print(f"[Law12RAG] Loaded {len(self.chunks)} chunks.")
        if use_embeddings:
            self._build_index()

    def _load_text(self, pdf_path):
        if pdf_path and Path(pdf_path).exists():
            try:
                import fitz

                doc = fitz.open(pdf_path)
                text = "".join(page.get_text() for page in doc)
                doc.close()
                print(f"[Law12RAG] Parsed PDF ({len(text)} chars)")

                # Find Law 12 section using min() — earliest valid match
                idx_law12 = text.find("Law 12")
                idx_fouls = text.upper().find("FOULS AND MISCONDUCT")
                candidates = [i for i in [idx_law12, idx_fouls] if i > 0]
                if candidates:
                    start = min(candidates)
                    for end_marker in ["Law 13", "LAW 13", "Law 14", "LAW 14"]:
                        end = text.find(end_marker, start + 100)
                        if end > start:
                            text = text[start:end]
                            print(
                                f"[Law12RAG] Extracted Law 12 section ({len(text)} chars)"
                            )
                            break
                    else:
                        text = text[start : start + 6000]
                        print(f"[Law12RAG] No end marker — using 6000 chars from start")
                return text
            except Exception as e:
                print(f"[Law12RAG] PDF parse error: {e}. Using hardcoded passages.")
        else:
            print("[Law12RAG] No PDF — using hardcoded Law 12 passages.")
        return LAW12_HARDCODED

    def _chunk(self, text: str, chunk_size: int):
        raw = re.split(r"\n{2,}", text)
        chunks, current = [], ""
        for part in raw:
            part = part.strip()
            if not part:
                continue
            if len(current) + len(part) < chunk_size:
                current += "\n" + part
            else:
                if current:
                    chunks.append(current.strip())
                current = part
        if current:
            chunks.append(current.strip())
        return [c for c in chunks if len(c) > 50]

    def _build_index(self):
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = self._model.encode(
                self.chunks, convert_to_numpy=True, show_progress_bar=False
            )
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            self.embeddings = emb / (norms + 1e-8)
            print("[Law12RAG] Embedding index built.")
        except ImportError:
            print("[Law12RAG] sentence-transformers unavailable — keyword retrieval.")
            self.use_embeddings = False

    def retrieve(self, query: str) -> str:
        if self.use_embeddings and self.embeddings is not None:
            q = self._model.encode([query], convert_to_numpy=True)
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
            scores = (self.embeddings @ q.T).squeeze()
            top_idx = np.argsort(scores)[::-1][: self.top_k]
            passages = [self.chunks[i] for i in top_idx]
        else:
            qw = set(query.lower().split())
            scored = sorted(
                self.chunks,
                key=lambda c: len(qw & set(c.lower().split())),
                reverse=True,
            )
            passages = scored[: self.top_k]

        ctx = "=== Relevant FIFA Law 12 Rules ===\n\n"
        for i, p in enumerate(passages, 1):
            ctx += f"[Rule {i}]\n{p}\n\n"
        return ctx.strip()

    def build_query(self, action_type: str) -> str:
        kw = {
            "Tackling": "tackle challenge from behind serious foul play red card",
            "Standing tackling": "tackle standing challenge careless reckless yellow card",
            "High leg": "high leg raised foot dangerous head endangers safety",
            "Holding": "holding opponent arms shirt DOGSO",
            "Pushing": "pushing opponent excessive force reckless",
            "Elbowing": "elbow violent conduct arm opponent not playing ball",
            "Challenge": "challenge aerial jump opponent contact",
            "Dive": "diving simulation feigning injury yellow card",
            "Dont know": "foul misconduct direct free kick",
        }
        return kw.get(action_type, "foul misconduct")


# ---------------------------------------------------------------------------
# MViT feature extractor (for RAG-ICL retrieval)
# ---------------------------------------------------------------------------


class MViTRetriever:
    """
    Extracts MViT-v2-S features from a PIL frame list and queries FAISS.
    Loaded lazily — only instantiated when strategy == rag_icl.
    """

    TARGET_FRAMES = 16
    FEAT_DIM = 768

    def __init__(self, index_path: str, meta_path: str, device: str = "cuda"):
        import faiss
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

        print("[MViTRetriever] Loading MViT-v2-S + FAISS index...")
        self.device = device

        # MViT backbone (head replaced with identity)
        weights = MViT_V2_S_Weights.DEFAULT
        model = mvit_v2_s(weights=weights)
        model.head = torch.nn.Identity()
        self.model = model.to(device).eval()
        self.transform = weights.transforms()

        # FAISS index + metadata
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.metadata = json.load(f)

        print(f"[MViTRetriever] Index has {self.index.ntotal} vectors.")

    @torch.no_grad()
    def _extract_feature(self, pil_frames: List[Image.Image]) -> np.ndarray:
        """
        frames_np : [T, H, W, C] uint8
        Returns   : [768] float32
        """
        # Convert PIL frames list to numpy array [T, H, W, C]
        frames_np = np.stack([np.array(p) for p in pil_frames])

        # [T, H, W, C] uint8 → [T, H, W, C] float → [1, C, T, H, W] for interpolate
        video = torch.from_numpy(frames_np.astype(np.float32))  # [T, H, W, C]
        video = video.permute(3, 0, 1, 2)  # [C, T, H, W]
        video = video.unsqueeze(0)  # [1, C, T, H, W]

        # Resample time dimension to 16 frames
        T = video.shape[2]
        if T != self.TARGET_FRAMES:
            video = F.interpolate(
                video,
                size=(self.TARGET_FRAMES, video.shape[3], video.shape[4]),
                mode="trilinear",
                align_corners=False,
            )  # [1, C, 16, H, W]

        # Convert back to uint8 [C, T, H, W] for MViT transforms
        video_uint8 = video.squeeze(0).clamp(0, 255).to(torch.uint8)  # [C, 16, H, W]

        # MViT transform expects [C, T, H, W] uint8
        input_tensor = (
            self.transform(video_uint8).unsqueeze(0).to(self.device)
        )  # [1, C, 16, H, W]

        feat = self.model(input_tensor).cpu().numpy().astype(np.float32).flatten()
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-8)

    def retrieve(self, live_frames: List[Image.Image], k: int = 3) -> str:
        """
        live_frames : PIL frames from the live (view 0) camera
        Returns     : formatted string of K precedents for prompt injection
        """
        feat = self._extract_feature(live_frames).reshape(1, -1)
        distances, indices = self.index.search(feat, k)

        examples_str = ""
        for rank, idx in enumerate(indices[0], 1):
            meta = self.metadata.get(str(idx), {})
            action = meta.get("action", "Unknown")
            sev = meta.get("severity", "Unknown")
            dist = float(distances[0][rank - 1])
            examples_str += (
                f"PRECEDENT {rank} "
                f"(visual similarity distance={dist:.3f}):\n"
                f"  Official referee decision: "
                f'{{"action": "{action}", "severity": "{sev}"}}\n\n'
            )
        return examples_str.strip()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ACTION_LIST_STR = "\n".join(f"  - {a}" for a in ACTION_CLASSES)
SEVERITY_LIST_STR = "\n".join(f"  - {s}" for s in SEVERITY_CLASSES)

SYSTEM_PROMPT = (
    "You are an expert football referee assistant. "
    "Analyze video frames from multiple camera angles and classify football "
    "foul incidents according to FIFA Laws of the Game. "
    "Always respond with ONLY a JSON object — no other text."
)

# ── Zero-shot ──────────────────────────────────────────────────────────────
ZERO_SHOT_TEMPLATE = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""

# ── Rule-grounded ──────────────────────────────────────────────────────────
RULE_GROUNDED_TEMPLATE = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

Based on the FIFA rules above, classify the incident:

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Rules summary:
- EXCESSIVE FORCE or endangering safety → RED CARD
- RECKLESS (disregard for opponent)     → YELLOW CARD
- CARELESS (lack of attention)          → No card but foul
- SIMULATION / DIVING                   → No offence + Yellow card

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""

# ── Static few-shot ────────────────────────────────────────────────────────
STATIC_EXAMPLES = """\
EXAMPLE 1:
Incident: Player lunges from behind, foot raised, full contact with opponent's leg.
Decision: {"action": "Tackling", "severity": "Red card"}
Reason: Tackle from behind with excessive force — serious foul play.

EXAMPLE 2:
Incident: Player extends elbow into opponent's face, ball not nearby.
Decision: {"action": "Elbowing", "severity": "Red card"}
Reason: Violent conduct regardless of ball proximity.

EXAMPLE 3:
Incident: Player falls dramatically after minimal contact.
Decision: {"action": "Dive", "severity": "No offence"}
Reason: Simulation — yellow card for unsporting behaviour.

EXAMPLE 4:
Incident: Player grabs opponent's shirt during a counterattack.
Decision: {"action": "Holding", "severity": "Yellow card"}
Reason: Reckless holding — disregards opponent's progress."""

STATIC_FEW_SHOT_TEMPLATE = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

Here are examples of correctly classified incidents:
{examples}

Now classify the incident shown in the video frames.

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""

# ── RAG-ICL (dynamic retrieval) ───────────────────────────────────────────
RAG_ICL_TEMPLATE = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

The following are VISUALLY SIMILAR fouls retrieved from a database of
previously judged incidents. The similarity is based on the motion pattern
of the live camera view. Use these as reference precedents:

{dynamic_examples}

Now classify the NEW incident shown in the video frames above.
Consider both the FIFA rules and the visual precedents when deciding.

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence citing rules or precedents>"}}"""


def build_prompt(
    strategy: str, n_views: int, law12_context: str = "", dynamic_examples: str = ""
) -> str:
    kw = dict(
        n_views=n_views,
        action_list=ACTION_LIST_STR,
        severity_list=SEVERITY_LIST_STR,
        law12_context=law12_context,
    )
    if strategy == "zero_shot":
        return ZERO_SHOT_TEMPLATE.format(**kw)
    elif strategy == "rule_grounded":
        return RULE_GROUNDED_TEMPLATE.format(**kw)
    elif strategy == "static_few_shot":
        return STATIC_FEW_SHOT_TEMPLATE.format(examples=STATIC_EXAMPLES, **kw)
    elif strategy == "rag_icl":
        return RAG_ICL_TEMPLATE.format(dynamic_examples=dynamic_examples, **kw)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def parse_response(text: str) -> Tuple[int, int]:
    text = re.sub(r"```json\s*|\s*```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return -1, -1
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        try:
            data = json.loads(match.group().replace("'", '"'))
        except Exception:
            return -1, -1

    a_str = data.get("action", "")
    s_str = data.get("severity", "")

    action_idx = next(
        (
            i
            for i, a in enumerate(ACTION_CLASSES)
            if a.lower() in a_str.lower() or a_str.lower() in a.lower()
        ),
        -1,
    )
    severity_idx = next(
        (
            i
            for i, s in enumerate(SEVERITY_CLASSES)
            if s.lower() in s_str.lower() or s_str.lower() in s.lower()
        ),
        -1,
    )
    return action_idx, severity_idx


# ---------------------------------------------------------------------------
# Qwen2.5-VL backend
# ---------------------------------------------------------------------------


class QwenVLBackend:
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

    def classify(self, frames_per_view: List[List[Image.Image]], prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        content = []
        for v_idx, frames in enumerate(frames_per_view):
            label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            content.append({"type": "text", "text": f"\n[{label}]"})
            for frame in frames:
                content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": f"\n\n{prompt}"})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
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


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


class VLMFoulClassifier:
    """
    Unified classifier supporting four strategies.
    For rag_icl, pass faiss_index_path + faiss_meta_path.
    """

    def __init__(
        self,
        backend: str = "qwen",
        strategy: str = "rag_icl",
        law12_pdf: str = None,
        frames_per_view: int = 4,
        faiss_index_path: str = None,
        faiss_meta_path: str = None,
        retrieval_k: int = 3,
    ):
        self.strategy = strategy
        self.frames_per_view = frames_per_view
        self.retrieval_k = retrieval_k

        # RAG over Law 12
        self.rag = Law12RAG(
            pdf_path=law12_pdf,
            top_k=3,
            use_embeddings=(backend != "gpt4o"),
        )

        # MViT retriever — only for rag_icl
        self.retriever = None
        if strategy == "rag_icl":
            if not faiss_index_path or not faiss_meta_path:
                raise ValueError(
                    "rag_icl requires --faiss_index_path and --faiss_meta_path"
                )
            if not Path(faiss_index_path).exists():
                raise FileNotFoundError(
                    f"FAISS index not found: {faiss_index_path}\n"
                    "Run build_faiss_index.py first."
                )
            self.retriever = MViTRetriever(
                index_path=faiss_index_path,
                meta_path=faiss_meta_path,
            )

        # VLM backend
        if backend == "qwen":
            self.backend = QwenVLBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def classify_action(
        self,
        frames_per_view: List[List[Image.Image]],
        action_hint: str = "Dont know",
    ) -> Tuple[int, int, str]:
        """
        frames_per_view : [[PIL, ...], [PIL, ...], ...]  one list per view
        action_hint     : used for RAG query (can be "Dont know")
        Returns         : (action_idx, severity_idx, raw_response)
        """
        # Law 12 RAG context (for all strategies except zero_shot)
        if self.strategy != "zero_shot":
            law12_ctx = self.rag.retrieve(self.rag.build_query(action_hint))
        else:
            law12_ctx = ""

        # Dynamic examples (rag_icl only)
        dynamic_examples = ""
        if self.strategy == "rag_icl" and self.retriever is not None:
            # Use live view (index 0) for retrieval
            live_frames = frames_per_view[0]
            dynamic_examples = self.retriever.retrieve(live_frames, k=self.retrieval_k)

        prompt = build_prompt(
            strategy=self.strategy,
            n_views=len(frames_per_view),
            law12_context=law12_ctx,
            dynamic_examples=dynamic_examples,
        )

        raw = self.backend.classify(frames_per_view, prompt)
        action_idx, severity_idx = parse_response(raw)
        return action_idx, severity_idx, raw
