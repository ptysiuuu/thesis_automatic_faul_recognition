"""
retrieval/law12_rag.py
======================
Retrieval-Augmented Generation over FIFA Law 12.
Parses the Laws PDF, chunks it, embeds with sentence-transformers,
and retrieves relevant passages for a given foul query.
"""

import re
import numpy as np
from pathlib import Path

LAW12_HARDCODED = """
=== FIFA Law 12: Fouls and Misconduct ===
CARELESS: lack of attention. No disciplinary sanction.
RECKLESS: disregard to the danger to an opponent. YELLOW CARD.
EXCESSIVE FORCE: far exceeded necessary force, endangered opponent. RED CARD.
RED CARD: serious foul play, violent conduct, DOGSO.
YELLOW CARD: reckless challenges, simulation/diving.
HIGH LEG: raising foot dangerously near opponent's head → RED CARD if excessive.
ELBOWING: violent conduct regardless of ball proximity → RED CARD.
DIVING/SIMULATION: feigning injury → YELLOW CARD.
TACKLING FROM BEHIND endangering safety → RED CARD.
RECKLESS TACKLE → YELLOW CARD. CARELESS TACKLE → free kick, no card.
HOLDING: direct free kick. Reckless → yellow. DOGSO → red.
PUSHING: direct free kick. Reckless → yellow. Excessive force → red.
"""

ACTION_KEYWORDS = {
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
                candidates = [i for i in [text.find("Law 12"),
                               text.upper().find("FOULS AND MISCONDUCT")] if i > 0]
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
            print("[Law12RAG] sentence-transformers unavailable — keyword retrieval.")
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
        return ACTION_KEYWORDS.get(action_type, "foul misconduct")
