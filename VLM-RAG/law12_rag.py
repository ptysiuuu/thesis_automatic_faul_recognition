"""
law12_rag.py
============
Retrieval-Augmented Generation over FIFA Law 12 PDF.

Pipeline:
  1. Parse Law 12 PDF into chunks
  2. Embed chunks with sentence-transformers (all-MiniLM-L6-v2, ~80MB, CPU-fast)
  3. Given a foul query (action_type, context), retrieve top-K relevant passages
  4. Return formatted context string for injection into VLM prompt

Dependencies:
  pip install pymupdf sentence-transformers numpy

Usage:
  rag = Law12RAG("law12.pdf")
  context = rag.retrieve("high leg tackle excessive force red card")
"""

import re
import json
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# FIFA Law 12 key passages — hardcoded fallback if PDF not available
# These are the most decision-relevant excerpts from Law 12 (2023/24 edition)
# ---------------------------------------------------------------------------

LAW12_HARDCODED = """
=== FIFA Law 12: Fouls and Misconduct ===

DIRECT FREE KICK:
A direct free kick is awarded if a player commits any of the following offences
against an opponent in a manner considered by the referee to be careless, reckless
or using excessive force:
- tackles or challenges an opponent
- kicks or attempts to kick an opponent
- jumps at an opponent
- charges an opponent
- strikes or attempts to strike an opponent (including head-butt)
- pushes an opponent
- holds an opponent
- impedes an opponent with contact
- bites or spits at an opponent
- trips or attempts to trip an opponent

CARELESS: the player has shown a lack of attention or consideration when making a
challenge or acted without precaution. No disciplinary sanction is needed.

RECKLESS: the player has acted with disregard to the danger to, or consequences
for, an opponent. The player must be cautioned (YELLOW CARD).

USING EXCESSIVE FORCE: the player has far exceeded the necessary use of force
and/or endangered the safety of an opponent. The player must be sent off (RED CARD).

YELLOW CARD - CAUTIONABLE OFFENCES:
A player is cautioned when committing the following offences:
- unsporting behaviour
- shows dissent by word or action
- persistently infringes the Laws of the Game
- delays the restart of play
- fails to respect the required distance at a free kick/corner kick/throw-in
- enters, re-enters or deliberately leaves the field without the referee's permission
- makes unauthorised marks on the field of play

RED CARD - SENDING-OFF OFFENCES:
A player is sent off if they commit any of the following offences:
- serious foul play (tackle or challenge that endangers the safety of an opponent
  or uses excessive force or brutality)
- violent conduct (uses or attempts to use excessive force or brutality against
  an opponent when not challenging for the ball, or against a team-mate,
  official, spectator or any other person)
- biting or spitting at anyone
- denying the opposing team a goal or an obvious goal-scoring opportunity (DOGSO)
  by a handball offence
- denying a goal or an obvious goal-scoring opportunity to an opponent moving
  towards the player's goal by an offence punishable by a free kick
- using offensive, insulting or abusive language and/or gestures
- receiving a second caution in the same match

SERIOUS FOUL PLAY:
A tackle or challenge that endangers the safety of an opponent or uses excessive
force or brutality must be sanctioned as serious foul play. Any player who lunges
at an opponent in challenging for the ball from the front, from the side or from
behind using one or both legs, with excessive force or endangers the safety of an
opponent is guilty of serious foul play.

HIGH FOOT / HIGH LEG:
A player who raises the foot dangerously close to an opponent's head or body,
where there is a risk of injury, may be penalised. If the player uses excessive
force or endangers the safety of an opponent, it is serious foul play (RED CARD).

HOLDING:
A player who holds an opponent is penalised with a direct free kick. If reckless,
a yellow card is shown. If it prevents an obvious goal-scoring opportunity (DOGSO),
a red card is shown.

PUSHING:
Pushing an opponent is a direct free kick offence. If reckless, yellow card.
If excessive force is used, red card (serious foul play).

ELBOWING:
Using the elbow against an opponent (violent conduct) is a red card offence,
even when not challenging for the ball.

CHALLENGE / TACKLING:
- Tackles from behind that endanger safety: RED CARD
- Tackles that are reckless: YELLOW CARD
- Careless tackles: free kick, no card

DIVING / SIMULATION:
A player who attempts to deceive the referee by feigning injury or pretending to
have been fouled (simulation / diving) must be cautioned for unsporting behaviour
(YELLOW CARD). This includes exaggerating the effect of contact.

CONTACT VS NO CONTACT:
For a foul to be penalised, physical contact is generally required (except for
impeding without contact and some handball situations). The referee must judge
whether contact occurred and its nature.
"""


class Law12RAG:
    """
    Retrieval system over FIFA Law 12.

    If a PDF path is provided and pymupdf is available, parses the PDF.
    Otherwise falls back to the hardcoded key passages above.

    Parameters
    ----------
    pdf_path : str or None
        Path to FIFA Laws of the Game PDF (Law 12 section).
        If None or file not found, uses hardcoded passages.
    chunk_size : int
        Approximate characters per chunk (default 400).
    top_k : int
        Number of passages to retrieve per query (default 3).
    use_embeddings : bool
        If True, uses sentence-transformers for semantic retrieval.
        If False, uses simple keyword overlap (faster, no GPU needed).
    """

    def __init__(
        self,
        pdf_path: str = None,
        chunk_size: int = 400,
        top_k: int = 3,
        use_embeddings: bool = True,
    ):
        self.top_k = top_k
        self.use_embeddings = use_embeddings
        self.chunks = []
        self.embeddings = None
        self._model = None

        # Load and chunk text
        text = self._load_text(pdf_path)
        self.chunks = self._chunk(text, chunk_size)
        print(f"[Law12RAG] Loaded {len(self.chunks)} chunks from Law 12.")

        # Build embedding index
        if use_embeddings:
            self._build_index()

    # ------------------------------------------------------------------
    def _load_text(self, pdf_path):
        if pdf_path and Path(pdf_path).exists():
            try:
                import fitz  # pymupdf
                doc = fitz.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                # Filter to Law 12 section if full Laws PDF
                if "Law 12" in text or "FOULS" in text.upper():
                    start = max(text.find("Law 12"), text.upper().find("FOULS AND MISCONDUCT"))
                    end = text.find("Law 13")
                    if start > 0 and end > start:
                        text = text[start:end]
                print(f"[Law12RAG] Parsed PDF: {pdf_path}")
                return text
            except ImportError:
                print("[Law12RAG] pymupdf not available, using hardcoded passages.")
            except Exception as e:
                print(f"[Law12RAG] PDF parse error: {e}. Using hardcoded passages.")
        else:
            print("[Law12RAG] No PDF provided, using hardcoded Law 12 passages.")
        return LAW12_HARDCODED

    def _chunk(self, text: str, chunk_size: int):
        # Split on section headers and paragraph breaks
        # Prefer splitting at double newlines or === headers
        raw = re.split(r'\n{2,}|(?====)', text)
        chunks = []
        current = ""
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
        # Filter very short chunks
        return [c for c in chunks if len(c) > 50]

    def _build_index(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = self._model.encode(
                self.chunks, convert_to_numpy=True, show_progress_bar=False
            )
            # L2 normalize for cosine similarity via dot product
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / (norms + 1e-8)
            print("[Law12RAG] Embedding index built.")
        except ImportError:
            print("[Law12RAG] sentence-transformers not available, using keyword retrieval.")
            self.use_embeddings = False

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> str:
        """
        Retrieve top-K relevant Law 12 passages for a query.

        Parameters
        ----------
        query : str
            Natural language query, e.g.
            "high leg tackle from behind excessive force"

        Returns
        -------
        str
            Formatted context string ready for prompt injection.
        """
        if self.use_embeddings and self.embeddings is not None:
            passages = self._semantic_retrieve(query)
        else:
            passages = self._keyword_retrieve(query)

        context = "=== Relevant FIFA Law 12 Rules ===\n\n"
        for i, p in enumerate(passages, 1):
            context += f"[Rule {i}]\n{p}\n\n"
        return context.strip()

    def _semantic_retrieve(self, query: str):
        q_emb = self._model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
        scores = (self.embeddings @ q_emb.T).squeeze()  # cosine similarity
        top_idx = np.argsort(scores)[::-1][:self.top_k]
        return [self.chunks[i] for i in top_idx]

    def _keyword_retrieve(self, query: str):
        query_words = set(query.lower().split())
        scored = []
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:self.top_k]]

    # ------------------------------------------------------------------
    def build_query(self, action_type: str, context_hint: str = "") -> str:
        """
        Build a retrieval query from predicted/candidate action type.

        Parameters
        ----------
        action_type : str
            One of the 8 action classes, or "unknown".
        context_hint : str
            Optional additional context ("from behind", "no contact", etc.)

        Returns
        -------
        str
            Query string for retrieval.
        """
        action_keywords = {
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
        base = action_keywords.get(action_type, "foul misconduct")
        return f"{base} {context_hint}".strip()
