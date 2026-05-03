"""
evaluate_ablation.py
====================
Unified ablation script for VLM-based foul classification.
Covers all three ablation rows + rag_icl via CLI flags.

ABLATION ROWS:
  Row 0 — static_few_shot baseline (current system, handcrafted examples)
  Row 1 — data_driven (mined medoid examples + class balance + severity prior)
  Row 2 — two_stage (data_driven examples, but action predicted first, then severity)
  Row 3 — rag_icl (MViT FAISS retrieval, dynamic examples)

Usage examples:

  # Row 0: baseline
  python evaluate_ablation.py \\
    --hdf5_path   /net/tscratch/people/plgaszos/SoccerNet_HDF5/Valid.hdf5 \\
    --annotations /net/tscratch/people/plgaszos/SoccerNet_Data/Valid/annotations.json \\
    --law12_pdf   /net/tscratch/people/plgaszos/SoccerNet_Data/law12.pdf \\
    --strategy    static_few_shot \\
    --output_dir  ablation_results/row0_baseline

  # Row 1: data-driven examples
  python evaluate_ablation.py \\
    --hdf5_path        /net/tscratch/people/plgaszos/SoccerNet_HDF5/Valid.hdf5 \\
    --annotations      /net/tscratch/people/plgaszos/SoccerNet_Data/Valid/annotations.json \\
    --train_hdf5       /net/tscratch/people/plgaszos/SoccerNet_HDF5/Train.hdf5 \\
    --train_annotations /net/tscratch/people/plgaszos/SoccerNet_Data/Train/annotations.json \\
    --law12_pdf        /net/tscratch/people/plgaszos/SoccerNet_Data/law12.pdf \\
    --faiss_index_path /net/tscratch/people/plgaszos/vlm_rag_icl/train_mvit_features.index \\
    --faiss_meta_path  /net/tscratch/people/plgaszos/vlm_rag_icl/train_mvit_metadata.json \\
    --strategy         data_driven \\
    --output_dir       ablation_results/row1_data_driven

  # Row 2: two-stage
  python evaluate_ablation.py \\
    ... (same as row 1) ... \\
    --strategy    two_stage \\
    --output_dir  ablation_results/row2_two_stage

  # Row 3: rag_icl
  python evaluate_ablation.py \\
    ... (same as row 1) ... \\
    --strategy    rag_icl \\
    --output_dir  ablation_results/row3_ragicl

  # Quick smoke test (50 samples)
  python evaluate_ablation.py ... --strategy data_driven --max_samples 50

  # Rebuild medoid cache (run once before row1/row2/row3)
  python evaluate_ablation.py \\
    --build_medoid_cache \\
    --train_hdf5        /net/tscratch/people/plgaszos/SoccerNet_HDF5/Train.hdf5 \\
    --train_annotations /net/tscratch/people/plgaszos/SoccerNet_Data/Train/annotations.json \\
    --faiss_index_path  /net/tscratch/people/plgaszos/vlm_rag_icl/train_mvit_features.index \\
    --faiss_meta_path   /net/tscratch/people/plgaszos/vlm_rag_icl/train_mvit_metadata.json \\
    --medoid_cache      /net/tscratch/people/plgaszos/vlm_rag_icl/medoid_cache.json
"""

import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from io import BytesIO
from collections import defaultdict, Counter

import torch
import h5py
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

ACTION_CLASSES = [
    "Tackling", "Standing tackling", "High leg", "Holding",
    "Pushing", "Elbowing", "Challenge", "Dive",
]
SEVERITY_CLASSES = ["No offence", "No card", "Yellow card", "Red card"]
ACTION_TO_IDX  = {a: i for i, a in enumerate(ACTION_CLASSES)}
SEVERITY_TO_IDX = {s: i for i, s in enumerate(SEVERITY_CLASSES)}

OFFENCE_SEVERITY_MAP = {
    ("No offence", ""): 0, ("No Offence", ""): 0,
    ("Offence", "1.0"): 1, ("Offence", "3.0"): 2, ("Offence", "5.0"): 3,
}

# Training-set severity priors (used for calibration hint in prompt).
# These are approximate; recomputed at runtime if --compute_priors is set.
SEVERITY_PRIOR_DEFAULT = {
    "No offence": 8,
    "No card": 20,
    "Yellow card": 58,
    "Red card": 14,
}


# ---------------------------------------------------------------------------
# Annotation loader (shared)
# ---------------------------------------------------------------------------

def load_annotations(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    samples = {}
    for action_id, action_data in data["Actions"].items():
        action_class   = action_data.get("Action class", "")
        offence_class  = action_data.get("Offence", "")
        severity_class = action_data.get("Severity", "")

        if action_class in {"Dont know", ""}:
            continue
        if offence_class in {"Between", ""} and action_class != "Dive":
            continue
        if severity_class in {"2.0", "4.0"} and action_class != "Dive" \
                and offence_class not in ("No offence", "No Offence"):
            continue
        if offence_class in {"Between", ""}:
            offence_class = "Offence"
        if severity_class in {"2.0", "4.0"}:
            severity_class = "1.0"

        key = (offence_class, severity_class)
        if key in OFFENCE_SEVERITY_MAP:
            severity_idx = OFFENCE_SEVERITY_MAP[key]
        elif offence_class in ("No offence", "No Offence"):
            severity_idx = 0
        else:
            continue

        action_idx = ACTION_TO_IDX.get(action_class, -1)
        if action_idx == -1:
            continue

        clips = [c["Url"].split("/")[-1] for c in action_data.get("Clips", [])]
        samples[action_id] = {
            "action":   action_idx,
            "severity": severity_idx,
            "clips":    clips,
        }
    return samples


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

def extract_keyframes(hdf5_file, action_key: str, clip_key: str,
                      n_frames: int = 4) -> List[Image.Image]:
    key = f"{action_key}/{clip_key}"
    if key not in hdf5_file:
        return []
    frames_np = hdf5_file[key][:]
    total = len(frames_np)
    if total < 2:
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    return [Image.fromarray(frames_np[i]) for i in indices]


def frames_to_base64(frames: List[Image.Image]) -> List[str]:
    result = []
    for img in frames:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        import base64
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return result


# ---------------------------------------------------------------------------
# Law 12 RAG
# ---------------------------------------------------------------------------

LAW12_HARDCODED = """
=== FIFA Law 12: Fouls and Misconduct ===
CARELESS: lack of attention. No disciplinary sanction.
RECKLESS: disregard for opponent's safety. YELLOW CARD.
EXCESSIVE FORCE: far exceeded necessary force, endangered opponent. RED CARD.
RED CARD: serious foul play, violent conduct, DOGSO.
YELLOW CARD: reckless challenges, simulation/diving.
HIGH LEG: raising foot dangerously near opponent's head → RED CARD if excessive.
ELBOWING: violent conduct regardless of ball proximity → RED CARD.
DIVING/SIMULATION: feigning injury → YELLOW CARD.
TACKLING FROM BEHIND endangering safety → RED CARD.
"""


class Law12RAG:
    def __init__(self, pdf_path: str = None, top_k: int = 3,
                 use_embeddings: bool = True):
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
                idx_law12 = text.find("Law 12")
                idx_fouls = text.upper().find("FOULS AND MISCONDUCT")
                candidates = [i for i in [idx_law12, idx_fouls] if i > 0]
                if candidates:
                    start = min(candidates)
                    for end_marker in ["Law 13", "LAW 13", "Law 14", "LAW 14"]:
                        end = text.find(end_marker, start + 100)
                        if end > start:
                            return text[start:end]
                    return text[start:start + 6000]
            except Exception as e:
                print(f"[Law12RAG] PDF error: {e}")
        return LAW12_HARDCODED

    def _chunk(self, text: str, chunk_size: int):
        lines = text.split("\n")
        chunks, current = [], ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(current) + len(line) < chunk_size:
                current += " " + line
            else:
                if current:
                    chunks.append(current.strip())
                current = line
        if current:
            chunks.append(current.strip())
        return [c for c in chunks if len(c) > 50]

    def _build_index(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = self._model.encode(self.chunks, convert_to_numpy=True,
                                     show_progress_bar=False)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            self.embeddings = emb / (norms + 1e-8)
        except ImportError:
            self.use_embeddings = False

    def retrieve(self, query: str) -> str:
        if self.use_embeddings and self.embeddings is not None:
            q = self._model.encode([query], convert_to_numpy=True)
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
            scores = (self.embeddings @ q.T).squeeze()
            top_idx = np.argsort(scores)[::-1][:self.top_k]
            passages = [self.chunks[i] for i in top_idx]
        else:
            qw = set(query.lower().split())
            passages = sorted(self.chunks,
                              key=lambda c: len(qw & set(c.lower().split())),
                              reverse=True)[:self.top_k]
        ctx = "=== Relevant FIFA Law 12 Rules ===\n\n"
        for i, p in enumerate(passages, 1):
            ctx += f"[Rule {i}]\n{p}\n\n"
        return ctx.strip()

    def build_query(self, action_type: str) -> str:
        kw = {
            "Tackling":          "tackle challenge from behind serious foul play red card",
            "Standing tackling": "tackle standing challenge careless reckless yellow card",
            "High leg":          "high leg raised foot dangerous head endangers safety",
            "Holding":           "holding opponent arms shirt DOGSO",
            "Pushing":           "pushing opponent excessive force reckless",
            "Elbowing":          "elbow violent conduct arm opponent not playing ball",
            "Challenge":         "challenge aerial jump opponent contact",
            "Dive":              "diving simulation feigning injury yellow card",
            "Dont know":         "foul misconduct direct free kick",
        }
        return kw.get(action_type, "foul misconduct")


# ---------------------------------------------------------------------------
# MViT feature extractor
# ---------------------------------------------------------------------------

class MViTExtractor:
    TARGET_FRAMES = 16
    FEAT_DIM = 768

    def __init__(self, device: str = "cuda"):
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        self.device = device
        weights = MViT_V2_S_Weights.DEFAULT
        model = mvit_v2_s(weights=weights)
        model.head = torch.nn.Identity()
        self.model = model.to(device).eval()
        self.transform = weights.transforms()

    @torch.no_grad()
    def extract_from_numpy(self, frames_np: np.ndarray) -> np.ndarray:
        """frames_np: [T, H, W, C] uint8"""
        T = len(frames_np)
        indices = np.linspace(0, T - 1, self.TARGET_FRAMES, dtype=int)
        sampled = frames_np[indices]
        video = torch.from_numpy(sampled).permute(0, 3, 1, 2).to(torch.uint8)
        inp = self.transform(video).unsqueeze(0).to(self.device)
        feat = self.model(inp).cpu().numpy().astype(np.float32).flatten()
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-8)

    @torch.no_grad()
    def extract_from_pil(self, pil_frames: List[Image.Image]) -> np.ndarray:
        frames_np = np.stack([np.array(f.convert("RGB")) for f in pil_frames])
        T = len(frames_np)
        indices = np.linspace(0, T - 1, self.TARGET_FRAMES, dtype=int).tolist()
        sampled = frames_np[indices]
        video = torch.from_numpy(sampled).permute(0, 3, 1, 2).to(torch.uint8)
        inp = self.transform(video).unsqueeze(0).to(self.device)
        feat = self.model(inp).cpu().numpy().astype(np.float32).flatten()
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-8)


# ---------------------------------------------------------------------------
# Medoid cache builder
# ---------------------------------------------------------------------------

def build_medoid_cache(train_hdf5_path: str, train_annotations_path: str,
                       faiss_index_path: str, faiss_meta_path: str,
                       output_path: str, n_frames_per_example: int = 2):
    """
    For each (action, severity) class combination, find the training sample
    whose MViT feature is closest to the class centroid (medoid).
    Saves N frames from that clip as base64 JPEGs for prompt injection.

    Output JSON structure:
    {
      "Tackling|Red card":   {"action": ..., "severity": ..., "frames_b64": [...]},
      "Elbowing|Red card":   {...},
      ...
    }
    """
    import faiss

    print("[MedoidCache] Loading FAISS index and metadata...")
    index = faiss.read_index(faiss_index_path)
    with open(faiss_meta_path) as f:
        metadata = json.load(f)

    # Group FAISS indices by (action, severity)
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx_str, meta in metadata.items():
        key = f"{meta['action']}|{meta['severity']}"
        groups[key].append(int(idx_str))

    print(f"[MedoidCache] Found {len(groups)} (action, severity) groups:")
    for k, v in sorted(groups.items()):
        print(f"  {k}: {len(v)} samples")

    # Reconstruct feature vectors from FAISS index
    all_feats = np.zeros((index.ntotal, MViTExtractor.FEAT_DIM), dtype=np.float32)
    for i in range(index.ntotal):
        faiss.downcast_index(index).reconstruct(i, all_feats[i])

    # Find medoid per group (index with minimum sum distance to all others)
    medoid_faiss_indices = {}
    for key, indices in groups.items():
        if len(indices) == 1:
            medoid_faiss_indices[key] = indices[0]
            continue
        feats = all_feats[indices]  # [N, 768]
        # Pairwise L2 distances
        dists = np.sum((feats[:, None] - feats[None, :]) ** 2, axis=-1)  # [N, N]
        sum_dists = dists.sum(axis=1)
        medoid_local = np.argmin(sum_dists)
        medoid_faiss_indices[key] = indices[medoid_local]

    # Load actual frames for each medoid from train HDF5
    print("[MedoidCache] Loading medoid frames from HDF5...")
    train_samples = load_annotations(train_annotations_path)

    # Build reverse map: faiss_idx → action_id
    faiss_to_action = {int(idx_str): meta["action_id"]
                       for idx_str, meta in metadata.items()}

    cache = {}
    with h5py.File(train_hdf5_path, "r", swmr=True) as hdf5:
        for key, faiss_idx in medoid_faiss_indices.items():
            action_id = faiss_to_action.get(faiss_idx)
            if action_id is None:
                print(f"  [WARN] No action_id for faiss_idx {faiss_idx}, key {key}")
                continue

            sample = train_samples.get(action_id)
            if sample is None:
                print(f"  [WARN] action_id {action_id} not in train_samples")
                continue

            # Load first (live) clip
            frames_b64 = []
            for clip_raw in sample["clips"][:1]:  # live view only
                clip_key = clip_raw.replace(".mp4", "")
                hdf5_key = f"action_{action_id}/{clip_key}"
                if hdf5_key not in hdf5:
                    continue
                frames_np = hdf5[hdf5_key][:]
                T = len(frames_np)
                if T < 2:
                    continue
                indices = np.linspace(0, T - 1, n_frames_per_example, dtype=int)
                for idx in indices:
                    img = Image.fromarray(frames_np[idx])
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    import base64
                    frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

            if frames_b64:
                action_str, severity_str = key.split("|")
                cache[key] = {
                    "action":     action_str,
                    "severity":   severity_str,
                    "action_id":  action_id,
                    "frames_b64": frames_b64,
                }
                print(f"  ✓ {key} → action_id={action_id}, {len(frames_b64)} frames")
            else:
                print(f"  ✗ {key} → no frames found")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache, f)
    print(f"[MedoidCache] Saved {len(cache)} entries → {output_path}")
    return cache


def load_medoid_cache(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Severity prior computation
# ---------------------------------------------------------------------------

def compute_severity_priors(train_annotations_path: str) -> dict:
    """Compute actual severity distribution from training set."""
    samples = load_annotations(train_annotations_path)
    counts = Counter(SEVERITY_CLASSES[s["severity"]] for s in samples.values())
    total = sum(counts.values())
    priors = {k: round(counts.get(k, 0) / total * 100) for k in SEVERITY_CLASSES}
    print(f"[Priors] Severity distribution: {priors}")
    return priors


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert football referee assistant. "
    "Analyze video frames from multiple camera angles and classify football "
    "foul incidents according to FIFA Laws of the Game. "
    "Always respond with ONLY a JSON object — no other text."
)

ACTION_LIST_STR   = "\n".join(f"  - {a}" for a in ACTION_CLASSES)
SEVERITY_LIST_STR = "\n".join(f"  - {s}" for s in SEVERITY_CLASSES)

STATIC_EXAMPLES = """\
EXAMPLE 1 — Tackling / Red card:
Incident: Player lunges from behind, foot raised, full contact with opponent's leg.
Decision: {"action": "Tackling", "severity": "Red card"}
Reason: Tackle from behind with excessive force — serious foul play.

EXAMPLE 2 — Elbowing / Red card:
Incident: Player extends elbow into opponent's face, ball not nearby.
Decision: {"action": "Elbowing", "severity": "Red card"}
Reason: Violent conduct regardless of ball proximity.

EXAMPLE 3 — Dive / No offence:
Incident: Player falls dramatically after minimal contact.
Decision: {"action": "Dive", "severity": "No offence"}
Reason: Simulation — yellow card for unsporting behaviour.

EXAMPLE 4 — Holding / Yellow card:
Incident: Player grabs opponent's shirt during a counterattack.
Decision: {"action": "Holding", "severity": "Yellow card"}
Reason: Reckless holding — disregards opponent's progress."""


def build_static_prompt(n_views: int, law12_context: str) -> str:
    return f"""\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

Here are examples of correctly classified incidents:
{STATIC_EXAMPLES}

Now classify the incident shown in the video frames.

ACTION TYPE (choose exactly one):
{ACTION_LIST_STR}

SEVERITY (choose exactly one):
{SEVERITY_LIST_STR}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""


def build_data_driven_prompt(n_views: int, law12_context: str,
                              mined_examples: str,
                              severity_priors: dict) -> str:
    prior_str = ", ".join(f"{k}: {v}%" for k, v in severity_priors.items())
    return f"""\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

The following are REAL examples from a database of officially judged incidents.
Each example shows the video frames and the correct referee decision:

{mined_examples}

SEVERITY CALIBRATION: In official match data the severity distribution is:
{prior_str}
Do not under-predict rare classes (No offence, Red card) — consider them seriously.

Now classify the NEW incident shown in the video frames above.

ACTION TYPE (choose exactly one):
{ACTION_LIST_STR}

SEVERITY (choose exactly one):
{SEVERITY_LIST_STR}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""


def build_two_stage_action_prompt(n_views: int, law12_context: str,
                                   mined_examples: str) -> str:
    """Stage 1: predict action type only."""
    return f"""\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

Reference examples:
{mined_examples}

Your task NOW: identify only the ACTION TYPE of the incident.

ACTION TYPE (choose exactly one):
{ACTION_LIST_STR}

Respond with ONLY this JSON:
{{"action": "<action type>", "reasoning": "<one sentence about the body movement>"}}"""


def build_two_stage_severity_prompt(n_views: int, law12_context: str,
                                     predicted_action: str,
                                     severity_examples: str,
                                     severity_priors: dict) -> str:
    """Stage 2: given predicted action, predict severity."""
    prior_str = ", ".join(f"{k}: {v}%" for k, v in severity_priors.items())
    return f"""\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

The action type has already been identified as: {predicted_action}

{law12_context}

Here are examples of how "{predicted_action}" incidents have been officially judged:
{severity_examples}

SEVERITY CALIBRATION: Overall severity distribution: {prior_str}

Given this is a "{predicted_action}", assess the severity:

SEVERITY (choose exactly one):
{SEVERITY_LIST_STR}

Rules:
- EXCESSIVE FORCE or endangering safety → RED CARD
- RECKLESS (disregard for opponent)     → YELLOW CARD
- CARELESS (lack of attention)          → No card but foul
- NO CONTACT / SIMULATION               → No offence

Respond with ONLY this JSON:
{{"severity": "<severity>", "reasoning": "<one sentence citing force level>"}}"""


def build_ragicl_prompt(n_views: int, law12_context: str,
                         dynamic_examples: str) -> str:
    return f"""\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

The following are VISUALLY SIMILAR fouls retrieved from a database of
previously judged incidents (retrieved by video motion similarity):

{dynamic_examples}

Now classify the NEW incident shown in the video frames above.

ACTION TYPE (choose exactly one):
{ACTION_LIST_STR}

SEVERITY (choose exactly one):
{SEVERITY_LIST_STR}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""


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
        (i for i, a in enumerate(ACTION_CLASSES)
         if a.lower() in a_str.lower() or a_str.lower() in a.lower()), -1)
    severity_idx = next(
        (i for i, s in enumerate(SEVERITY_CLASSES)
         if s.lower() in s_str.lower() or s_str.lower() in s.lower()), -1)
    return action_idx, severity_idx


def parse_action_only(text: str) -> int:
    text = re.sub(r"```json\s*|\s*```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return -1
    try:
        data = json.loads(match.group())
    except Exception:
        return -1
    a_str = data.get("action", "")
    return next(
        (i for i, a in enumerate(ACTION_CLASSES)
         if a.lower() in a_str.lower() or a_str.lower() in a.lower()), -1)


def parse_severity_only(text: str) -> int:
    text = re.sub(r"```json\s*|\s*```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return -1
    try:
        data = json.loads(match.group())
    except Exception:
        return -1
    s_str = data.get("severity", "")
    return next(
        (i for i, s in enumerate(SEVERITY_CLASSES)
         if s.lower() in s_str.lower() or s_str.lower() in s.lower()), -1)


# ---------------------------------------------------------------------------
# Qwen2.5-VL backend
# ---------------------------------------------------------------------------

class QwenVLBackend:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"[QwenVL] Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True,
        )
        self.model.eval()
        print("[QwenVL] Ready.")

    def classify(self, frames_per_view: List[List[Image.Image]],
                 prompt: str,
                 extra_images: List[Image.Image] = None) -> str:
        """
        extra_images: additional PIL images prepended before the main views
                      (used for mined medoid examples in data_driven mode)
        """
        from qwen_vl_utils import process_vision_info

        content = []

        # Inject mined example images if provided
        if extra_images:
            content.append({"type": "text", "text": "[Reference examples from training data:]"})
            for img in extra_images:
                content.append({"type": "image", "image": img})

        # Main multi-view frames
        for v_idx, frames in enumerate(frames_per_view):
            label = "Live camera" if v_idx == 0 else f"Replay {v_idx}"
            content.append({"type": "text", "text": f"\n[{label}]"})
            for frame in frames:
                content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": f"\n\n{prompt}"})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input], images=img_inputs, videos=vid_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=512,
                do_sample=False, temperature=None, top_p=None,
            )
        generated = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# MViT FAISS retriever (for rag_icl)
# ---------------------------------------------------------------------------

class MViTRetriever:
    TARGET_FRAMES = 16
    FEAT_DIM = 768

    def __init__(self, index_path: str, meta_path: str, device: str = "cuda"):
        import faiss
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        print("[MViTRetriever] Loading MViT-v2-S + FAISS index...")
        self.device = device
        weights = MViT_V2_S_Weights.DEFAULT
        model = mvit_v2_s(weights=weights)
        model.head = torch.nn.Identity()
        self.model = model.to(device).eval()
        self.transform = weights.transforms()
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.metadata = json.load(f)
        print(f"[MViTRetriever] Index has {self.index.ntotal} vectors.")

    @torch.no_grad()
    def _extract_feature(self, pil_frames: List[Image.Image]) -> np.ndarray:
        frames_np = np.stack([np.array(f.convert("RGB")) for f in pil_frames])
        T = len(frames_np)
        indices = np.linspace(0, T - 1, self.TARGET_FRAMES, dtype=int).tolist()
        sampled = frames_np[indices]
        video = torch.from_numpy(sampled).permute(0, 3, 1, 2).to(torch.uint8)
        inp = self.transform(video).unsqueeze(0).to(self.device)
        feat = self.model(inp).cpu().numpy().astype(np.float32).flatten()
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-8)

    def retrieve(self, live_frames: List[Image.Image], k: int = 3) -> str:
        """Diverse retrieval: one example per unique action class."""
        feat = self._extract_feature(live_frames).reshape(1, -1).astype(np.float32)
        n_candidates = min(self.index.ntotal, 50)
        distances, indices = self.index.search(feat, n_candidates)

        seen_actions = {}
        for dist, idx in zip(distances[0], indices[0]):
            meta = self.metadata.get(str(idx), {})
            action = meta.get("action", "Unknown")
            if action not in seen_actions:
                seen_actions[action] = (float(dist), meta)
            if len(seen_actions) >= k:
                break

        examples_str = ""
        for rank, (action, (dist, meta)) in enumerate(seen_actions.items(), 1):
            sev = meta.get("severity", "Unknown")
            examples_str += (
                f"PRECEDENT {rank} (visual distance={dist:.3f}):\n"
                f'  Decision: {{"action": "{action}", "severity": "{sev}"}}\n\n'
            )
        return examples_str.strip()


# ---------------------------------------------------------------------------
# Mined examples builder (for data_driven and two_stage)
# ---------------------------------------------------------------------------

def build_mined_examples_text(medoid_cache: dict,
                               n_per_class: int = 1) -> Tuple[str, List[Image.Image]]:
    """
    Build text description + list of PIL images from medoid cache.
    Returns (examples_text, list_of_pil_images).
    The PIL images are injected into the VLM prompt as actual visual examples.
    """
    import base64

    examples_text = ""
    example_images = []
    shown = 0
    for key, entry in sorted(medoid_cache.items()):
        action   = entry["action"]
        severity = entry["severity"]
        frames_b64 = entry["frames_b64"][:n_per_class]

        examples_text += (
            f"EXAMPLE {shown + 1} — {action} / {severity}:\n"
            f'  Decision: {{"action": "{action}", "severity": "{severity}"}}\n\n'
        )
        for b64 in frames_b64:
            img_bytes = base64.b64decode(b64)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            example_images.append(img)
        shown += 1

    return examples_text.strip(), example_images


def build_severity_examples_for_action(medoid_cache: dict,
                                        action: str) -> Tuple[str, List[Image.Image]]:
    """
    For two-stage stage 2: get all severity examples for a given action.
    """
    import base64

    examples_text = ""
    example_images = []
    rank = 1
    for key, entry in sorted(medoid_cache.items()):
        if entry["action"] != action:
            continue
        severity = entry["severity"]
        frames_b64 = entry["frames_b64"][:1]
        examples_text += (
            f"EXAMPLE {rank} — {action} / {severity}:\n"
            f'  Decision: {{"action": "{action}", "severity": "{severity}"}}\n\n'
        )
        for b64 in frames_b64:
            img_bytes = base64.b64decode(b64)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            example_images.append(img)
        rank += 1

    return examples_text.strip(), example_images


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true_a, y_pred_a, y_true_s, y_pred_s) -> dict:
    valid = [i for i in range(len(y_pred_a))
             if y_pred_a[i] != -1 and y_pred_s[i] != -1]
    n_valid = len(valid)
    n_total = len(y_pred_a)
    if n_valid == 0:
        return {"error": "No valid predictions"}

    ya_true = [y_true_a[i] for i in valid]
    ya_pred = [y_pred_a[i] for i in valid]
    ys_true = [y_true_s[i] for i in valid]
    ys_pred = [y_pred_s[i] for i in valid]

    return {
        "n_total":               n_total,
        "n_valid":               n_valid,
        "parse_rate":            n_valid / n_total * 100,
        "accuracy_action":       accuracy_score(ya_true, ya_pred) * 100,
        "balanced_acc_action":   balanced_accuracy_score(ya_true, ya_pred) * 100,
        "accuracy_severity":     accuracy_score(ys_true, ys_pred) * 100,
        "balanced_acc_severity": balanced_accuracy_score(ys_true, ys_pred) * 100,
        "leaderboard_value":     (balanced_accuracy_score(ya_true, ya_pred) +
                                  balanced_accuracy_score(ys_true, ys_pred)) / 2 * 100,
        "confusion_action":      confusion_matrix(
            ya_true, ya_pred, labels=list(range(len(ACTION_CLASSES)))).tolist(),
        "confusion_severity":    confusion_matrix(
            ys_true, ys_pred, labels=list(range(len(SEVERITY_CLASSES)))).tolist(),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Strategy: {args.strategy}")
    print(f"{'='*60}")

    # Load eval annotations
    print("Loading annotations...")
    samples = load_annotations(args.annotations)
    if args.max_samples:
        sample_ids = list(samples.keys())[:args.max_samples]
        samples = {k: samples[k] for k in sample_ids}
    print(f"Evaluating on {len(samples)} samples.")

    # Law 12 RAG
    rag = Law12RAG(pdf_path=args.law12_pdf, top_k=3, use_embeddings=True)

    # Severity priors
    if args.train_annotations and Path(args.train_annotations).exists():
        severity_priors = compute_severity_priors(args.train_annotations)
    else:
        severity_priors = SEVERITY_PRIOR_DEFAULT
        print(f"[Priors] Using default priors: {severity_priors}")

    # Load medoid cache for data_driven / two_stage
    medoid_cache = None
    if args.strategy in ("data_driven", "two_stage"):
        if not args.medoid_cache or not Path(args.medoid_cache).exists():
            raise FileNotFoundError(
                f"Medoid cache not found: {args.medoid_cache}\n"
                "Run with --build_medoid_cache first."
            )
        medoid_cache = load_medoid_cache(args.medoid_cache)
        print(f"[MedoidCache] Loaded {len(medoid_cache)} entries.")

    # Pre-build mined examples text + images (data_driven only, not two_stage)
    mined_examples_text = None
    mined_example_images = None
    if args.strategy == "data_driven" and medoid_cache:
        mined_examples_text, mined_example_images = build_mined_examples_text(
            medoid_cache, n_per_class=1)

    # FAISS retriever (rag_icl)
    retriever = None
    if args.strategy == "rag_icl":
        if not args.faiss_index_path or not args.faiss_meta_path:
            raise ValueError("rag_icl requires --faiss_index_path and --faiss_meta_path")
        retriever = MViTRetriever(args.faiss_index_path, args.faiss_meta_path)

    # VLM backend (loaded once, reused)
    backend = QwenVLBackend(model_name=args.model_name)

    # Eval loop
    y_true_a, y_pred_a = [], []
    y_true_s, y_pred_s = [], []
    predictions = {}

    with h5py.File(args.hdf5_path, "r", swmr=True) as hdf5:
        for action_id, sample in tqdm(samples.items(), desc=f"  [{args.strategy}]"):
            action_key = f"action_{action_id}"

            # Load frames
            frames_per_view_list = []
            for clip_raw in sample["clips"][:4]:
                clip_key = clip_raw.replace(".mp4", "")
                frames = extract_keyframes(hdf5, action_key, clip_key,
                                           n_frames=args.frames_per_view)
                if frames:
                    frames_per_view_list.append(frames)

            if not frames_per_view_list:
                y_true_a.append(sample["action"]); y_pred_a.append(-1)
                y_true_s.append(sample["severity"]); y_pred_s.append(-1)
                continue

            action_hint = ACTION_CLASSES[sample["action"]]  # used for RAG query
            law12_ctx = rag.retrieve(rag.build_query(action_hint))

            try:
                # ── Row 0: static_few_shot ────────────────────────────────
                if args.strategy == "static_few_shot":
                    prompt = build_static_prompt(
                        n_views=len(frames_per_view_list),
                        law12_context=law12_ctx,
                    )
                    raw = backend.classify(frames_per_view_list, prompt)
                    act_idx, sev_idx = parse_response(raw)

                # ── Row 1: data_driven ────────────────────────────────────
                elif args.strategy == "data_driven":
                    prompt = build_data_driven_prompt(
                        n_views=len(frames_per_view_list),
                        law12_context=law12_ctx,
                        mined_examples=mined_examples_text,
                        severity_priors=severity_priors,
                    )
                    raw = backend.classify(
                        frames_per_view_list, prompt,
                        extra_images=mined_example_images,
                    )
                    act_idx, sev_idx = parse_response(raw)

                # ── Row 2: two_stage ──────────────────────────────────────
                elif args.strategy == "two_stage":
                    # Stage 1: predict action
                    all_mined_text, all_mined_imgs = build_mined_examples_text(
                        medoid_cache, n_per_class=1)
                    stage1_prompt = build_two_stage_action_prompt(
                        n_views=len(frames_per_view_list),
                        law12_context=law12_ctx,
                        mined_examples=all_mined_text,
                    )
                    raw_stage1 = backend.classify(
                        frames_per_view_list, stage1_prompt,
                        extra_images=all_mined_imgs,
                    )
                    act_idx = parse_action_only(raw_stage1)
                    if act_idx == -1:
                        act_idx_str = "Dont know"
                    else:
                        act_idx_str = ACTION_CLASSES[act_idx]

                    # Stage 2: predict severity conditioned on action
                    sev_examples_text, sev_example_imgs = build_severity_examples_for_action(
                        medoid_cache, action=act_idx_str)
                    stage2_law12 = rag.retrieve(rag.build_query(act_idx_str))
                    stage2_prompt = build_two_stage_severity_prompt(
                        n_views=len(frames_per_view_list),
                        law12_context=stage2_law12,
                        predicted_action=act_idx_str,
                        severity_examples=sev_examples_text,
                        severity_priors=severity_priors,
                    )
                    raw_stage2 = backend.classify(
                        frames_per_view_list, stage2_prompt,
                        extra_images=sev_example_imgs if sev_example_imgs else None,
                    )
                    sev_idx = parse_severity_only(raw_stage2)
                    raw = f"STAGE1: {raw_stage1}\nSTAGE2: {raw_stage2}"

                # ── Row 3: rag_icl ────────────────────────────────────────
                elif args.strategy == "rag_icl":
                    dynamic_examples = retriever.retrieve(
                        frames_per_view_list[0], k=args.retrieval_k)
                    prompt = build_ragicl_prompt(
                        n_views=len(frames_per_view_list),
                        law12_context=law12_ctx,
                        dynamic_examples=dynamic_examples,
                    )
                    raw = backend.classify(frames_per_view_list, prompt)
                    act_idx, sev_idx = parse_response(raw)

                else:
                    raise ValueError(f"Unknown strategy: {args.strategy}")

            except Exception as e:
                print(f"  Error on {action_id}: {e}")
                act_idx, sev_idx, raw = -1, -1, str(e)

            y_true_a.append(sample["action"]); y_pred_a.append(act_idx)
            y_true_s.append(sample["severity"]); y_pred_s.append(sev_idx)
            predictions[action_id] = {
                "true_action":   sample["action"],
                "pred_action":   act_idx,
                "true_severity": sample["severity"],
                "pred_severity": sev_idx,
                "raw_response":  raw,
            }

    metrics = compute_metrics(y_true_a, y_pred_a, y_true_s, y_pred_s)
    metrics["strategy"] = args.strategy
    metrics["predictions"] = predictions

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Summary (no predictions)
    summary = {k: v for k, v in metrics.items()
               if k not in ("predictions", "confusion_action", "confusion_severity")}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS — {args.strategy}")
    print(f"{'='*60}")
    print(f"  Parse rate:    {metrics.get('parse_rate', 0):.1f}%")
    print(f"  Action BA:     {metrics.get('balanced_acc_action', 0):.2f}%")
    print(f"  Severity BA:   {metrics.get('balanced_acc_severity', 0):.2f}%")
    print(f"  Leaderboard:   {metrics.get('leaderboard_value', 0):.4f}")
    print(f"\nSaved to {output_dir}/")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VLM foul classification ablation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode ──────────────────────────────────────────────────────────────
    parser.add_argument("--build_medoid_cache", action="store_true",
        help="Build medoid cache from training data and exit. "
             "Run once before using data_driven/two_stage strategies.")

    # ── Strategy ─────────────────────────────────────────────────────────
    parser.add_argument("--strategy",
        choices=["static_few_shot", "data_driven", "two_stage", "rag_icl"],
        default="static_few_shot",
        help="Ablation row to evaluate.")

    # ── Data paths ───────────────────────────────────────────────────────
    parser.add_argument("--hdf5_path",
        help="Path to eval HDF5 (Valid.hdf5 or Test.hdf5)")
    parser.add_argument("--annotations",
        help="Path to eval annotations.json")
    parser.add_argument("--train_hdf5",
        help="Path to Train.hdf5 (needed for medoid cache building)")
    parser.add_argument("--train_annotations",
        help="Path to Train/annotations.json (needed for priors + medoid cache)")
    parser.add_argument("--law12_pdf",
        default=None,
        help="Path to FIFA Laws PDF (Law 12). Uses hardcoded fallback if omitted.")

    # ── FAISS / medoid ────────────────────────────────────────────────────
    parser.add_argument("--faiss_index_path", default=None,
        help="Path to FAISS index (required for data_driven, two_stage, rag_icl)")
    parser.add_argument("--faiss_meta_path", default=None,
        help="Path to FAISS metadata JSON")
    parser.add_argument("--medoid_cache",
        default="/net/tscratch/people/plgaszos/vlm_rag_icl/medoid_cache.json",
        help="Path to medoid cache JSON (built with --build_medoid_cache)")

    # ── Model ────────────────────────────────────────────────────────────
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--frames_per_view", type=int, default=4)
    parser.add_argument("--retrieval_k", type=int, default=3,
        help="Number of examples to retrieve for rag_icl")

    # ── Output ───────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", default="ablation_results/output")
    parser.add_argument("--max_samples", type=int, default=None,
        help="Limit samples for quick testing")

    args = parser.parse_args()

    # ── Medoid cache build mode ───────────────────────────────────────────
    if args.build_medoid_cache:
        if not all([args.train_hdf5, args.train_annotations,
                    args.faiss_index_path, args.faiss_meta_path]):
            parser.error(
                "--build_medoid_cache requires: "
                "--train_hdf5, --train_annotations, "
                "--faiss_index_path, --faiss_meta_path"
            )
        build_medoid_cache(
            train_hdf5_path=args.train_hdf5,
            train_annotations_path=args.train_annotations,
            faiss_index_path=args.faiss_index_path,
            faiss_meta_path=args.faiss_meta_path,
            output_path=args.medoid_cache,
        )
        print("Medoid cache built. Re-run without --build_medoid_cache to evaluate.")
        return

    # ── Evaluation mode ───────────────────────────────────────────────────
    if not args.hdf5_path or not args.annotations:
        parser.error("Evaluation requires --hdf5_path and --annotations")

    evaluate(args)


if __name__ == "__main__":
    main()
