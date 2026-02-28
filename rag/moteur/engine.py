from __future__ import annotations

import os
import re
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List
from uuid import uuid4
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from mistralai import Mistral

from rag.preparation.indexing import (
    SplitConfig,
    build_faiss_index,
    events_to_documents,
    load_faiss_index,
    load_processed_events,
    split_documents,
)
from rag.moteur.rag_chain import _mistral_chat, build_context


_tz_name = os.environ.get("P9_TZ", "Europe/Paris")
try:
    PARIS_TZ = ZoneInfo(_tz_name)
except Exception:
    # On Windows, the IANA tz database may be missing unless `tzdata` is installed.
    # Fallback to UTC to avoid crashing at import time.
    PARIS_TZ = timezone.utc


@dataclass
class RebuildStats:
    rows: int
    docs: int
    chunks: int
    index_dir: str


def _parse_dt(s: str | None) -> datetime | None:
    """Parse an ISO datetime string into an aware datetime (if possible)."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _time_intent(question: str) -> str:
    """Return one of: future | past | neutral."""
    q = (question or "").lower()

    future_markers = [
        "ce week-end",
        "ce weekend",
        "week-end",
        "weekend",
        "demain",
        "aujourd",
        "ce soir",
        "à venir",
        "a venir",
        "prochain",
        "cette semaine",
        "la semaine prochaine",
        "ce mois",
        "le mois prochain",
    ]
    if any(m in q for m in future_markers):
        return "future"

    # explicit years or clear past markers
    if re.search(r"\b(19|20)\d{2}\b", q):
        return "past"

    past_markers = [
        "l'an dernier",
        "l’année dernière",
        "l'annee derniere",
        "passé",
        "passe",
        "historique",
        "il y a ",
    ]
    if any(m in q for m in past_markers):
        return "past"

    return "neutral"


def _next_weekend_window(now: datetime) -> tuple[datetime, datetime]:
    """Compute next weekend window [Sat 00:00, Sun 23:59:59] in `now` tz."""
    # weekday: Monday=0 ... Sunday=6 ; Saturday=5
    days_until_sat = (5 - now.weekday()) % 7
    sat = (now + timedelta(days=days_until_sat)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    sun = sat + timedelta(days=1)
    end = sun.replace(hour=23, minute=59, second=59, microsecond=0)
    return sat, end


def _filter_docs_by_window(
    docs: list[Any], start_dt: datetime, end_dt: datetime
) -> list[Any]:
    """Keep docs whose [event_start,event_end] overlaps the window."""
    kept: list[Any] = []
    for d in docs:
        md = getattr(d, "metadata", None) or {}
        s = _parse_dt(md.get("event_start"))
        e = _parse_dt(md.get("event_end")) or s
        if s is None and e is None:
            continue
        if e is None:
            e = s
        if s is None:
            s = e

        # Overlap: [s,e] intersects [start_dt,end_dt]
        if e >= start_dt and s <= end_dt:
            kept.append(d)
    return kept


class RagEngine:
    """Holds embeddings + FAISS index in memory for fast /ask.

    Also keeps a small in-memory conversation history (ring buffer) for:
    - continuity on follow-up questions (prompt injection of last N turns)
    - simple thumbs up / thumbs down feedback

    Can rebuild the FAISS index on demand.
    """

    def __init__(
        self,
        index_dir: Path,
        processed_csv: Path,
        embeddings: Any,
        chat_model: str,
        k_default: int = 6,
        split_cfg: SplitConfig = SplitConfig(),
        history_size: int = 10,
        history_prompt_turns: int = 3,
    ) -> None:
        self.index_dir = index_dir
        self.processed_csv = processed_csv
        self.embeddings = embeddings
        self.chat_model = chat_model
        self.k_default = k_default
        self.split_cfg = split_cfg

        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._history_prompt_turns = history_prompt_turns

        self._lock = threading.RLock()
        self._vs = None  # FAISS vectorstore (LangChain wrapper)

        load_dotenv()
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY manquant. Vérifie ton fichier .env.")
        self._client = Mistral(api_key=api_key)

    def load(self) -> None:
        with self._lock:
            self._vs = load_faiss_index(self.index_dir, self.embeddings)

    def ask(self, question: str, k: int | None = None) -> dict[str, Any]:
        q = (question or "").strip()
        if not q:
            raise ValueError("Question vide.")

        kk = int(k or self.k_default)

        # --- Retrieval (retrieve more than kk, then filter by time intent) ---
        retrieve_mult = int(os.environ.get("P9_RETRIEVE_MULT", "4"))
        max_retrieve = int(os.environ.get("P9_MAX_RETRIEVE", "50"))
        k_retrieve = min(max_retrieve, max(kk, 1) * max(1, retrieve_mult))

        with self._lock:
            if self._vs is None:
                raise RuntimeError(
                    "Index non chargé. Lance /rebuild ou vérifie P9_INDEX_DIR."
                )
            docs = self._vs.similarity_search(q, k=k_retrieve)

        # --- Time intent handling ---
        now_paris = datetime.now(PARIS_TZ)
        intent = _time_intent(q)
        time_guidance = ""

        if intent == "future":
            future_days = int(os.environ.get("P9_FUTURE_WINDOW_DAYS", "30"))
            ql = q.lower()

            if "week-end" in ql or "weekend" in ql:
                w_start, w_end = _next_weekend_window(now_paris)
                filtered = _filter_docs_by_window(docs, w_start, w_end)

                if filtered:
                    docs = filtered[:kk]
                    time_guidance = (
                        "La question vise le futur (ex: 'ce week-end'). "
                        f"Ne propose que des événements dans la fenêtre {w_start.isoformat()} → {w_end.isoformat()}."
                    )
                else:
                    # fallback: broaden to next N days
                    broad_end = now_paris + timedelta(days=future_days)
                    filtered2 = _filter_docs_by_window(docs, now_paris, broad_end)
                    if filtered2:
                        docs = filtered2[:kk]
                        time_guidance = (
                            "La question vise le futur. Aucun événement trouvé strictement sur le prochain week-end; "
                            f"je propose des événements à venir dans les {future_days} prochains jours "
                            f"({now_paris.isoformat()} → {broad_end.isoformat()})."
                        )
                    else:
                        docs = docs[:kk]
                        time_guidance = (
                            "La question vise le futur, mais je n'ai trouvé aucun événement à venir dans le corpus "
                            f"sur les {future_days} prochains jours. Je fournis les éléments les plus proches disponibles "
                            "(peuvent être passés)."
                        )
            else:
                broad_end = now_paris + timedelta(days=future_days)
                filtered = _filter_docs_by_window(docs, now_paris, broad_end)
                if filtered:
                    docs = filtered[:kk]
                    time_guidance = (
                        "La question vise le futur. "
                        f"Ne propose que des événements à venir dans la fenêtre {now_paris.isoformat()} → {broad_end.isoformat()}."
                    )
                else:
                    docs = docs[:kk]
                    time_guidance = (
                        "La question vise le futur, mais je n'ai trouvé aucun événement à venir dans le corpus "
                        f"sur les {future_days} prochains jours. Je fournis les éléments les plus proches disponibles "
                        "(peuvent être passés)."
                    )

        elif intent == "past":
            # If a year is mentioned, filter to that year.
            m = re.search(r"\b(19|20)\d{2}\b", q)
            if m:
                year = int(m.group(0))
                start = datetime(year, 1, 1, tzinfo=PARIS_TZ)
                end = datetime(year, 12, 31, 23, 59, 59, tzinfo=PARIS_TZ)
                filtered = _filter_docs_by_window(docs, start, end)
                docs = filtered[:kk] if filtered else docs[:kk]
                time_guidance = (
                    "La question porte sur le passé. "
                    f"Privilégie des événements de l'année {year} (fenêtre {start.isoformat()} → {end.isoformat()})."
                )
            else:
                docs = docs[:kk]
                time_guidance = "La question porte sur le passé. Tu peux proposer des événements passés correspondant à la demande."

        else:
            docs = docs[:kk]
            time_guidance = (
                "La question ne précise pas clairement passé/futur. "
                "Si des dates apparaissent, indique clairement si chaque événement est passé ou à venir."
            )

        # --- Retrieved contexts (for RAGAS evaluation) ---
        retrieved_contexts = [getattr(d, "page_content", "") for d in docs]

        # --- Context building ---
        context_text, sources = build_context(docs)

        # --- Conversation continuity (last N turns) ---
        history_text = self._format_history_for_prompt()

        # --- Bonus prompt: reference date + clear time behavior ---
        reference_line = f"DATE_DE_REFERENCE (Europe/Paris): {now_paris.isoformat()}"
        intent_line = f"INTENTION_TEMPORELLE: {intent}"

        system = (
            "Tu es un assistant culturel. "
            "Règles: "
            "1) Réponds uniquement avec les informations du CONTEXTE (événements/sources). "
            "2) Tu peux utiliser l'HISTORIQUE uniquement pour comprendre les questions de suivi "
            "(ex: 'et demain', 'dans la même ville', préférences), mais jamais comme source factuelle d'événements. "
            "3) Respecte la DATE_DE_REFERENCE pour interpréter les expressions relatives (ex: 'ce week-end', 'demain'). "
            "4) Applique les CONTRAINTES_TEMPS si elles sont fournies. "
            "5) Propose au maximum 5 recommandations. "
            "6) Pour chaque recommandation, donne le titre, les date(s), la ville et l'URL. "
            "7) Termine par 'Sources:' avec les identifiants [S1], [S2]..."
        )

        user_parts: list[str] = [reference_line, intent_line]
        if time_guidance:
            user_parts.append(f"CONTRAINTES_TEMPS: {time_guidance}")
        if history_text:
            user_parts.append(f"HISTORIQUE (dern. échanges):\n{history_text}")
        user_parts.append(f"QUESTION:\n{q}")
        user_parts.append(f"CONTEXTE:\n{context_text}")
        user_parts.append("RÉPONSE:")
        user = "\n\n".join(user_parts)

        answer = _mistral_chat(
            self._client, model=self.chat_model, system=system, user=user
        )

        turn_id = uuid4().hex
        result: Dict[str, Any] = {
            "turn_id": turn_id,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "question": q,
            "answer": answer,
            "sources": sources,
            "k": kk,
            "model": self.chat_model,
            "rating": None,
            "retrieved_contexts": retrieved_contexts,
        }

        # --- Store a light version in history (avoid huge memory + /history payloads) ---
        history_item = dict(result)
        history_item.pop("retrieved_contexts", None)
        self._log_history(history_item)

        return result

    def rebuild(self) -> RebuildStats:
        """Rebuild FAISS index from processed CSV and reload in memory."""
        with self._lock:
            df = load_processed_events(self.processed_csv)
            docs = events_to_documents(df)
            chunks = split_documents(docs, self.split_cfg)

            build_faiss_index(chunks, self.embeddings, self.index_dir)
            self._vs = load_faiss_index(self.index_dir, self.embeddings)

            return RebuildStats(
                rows=len(df),
                docs=len(docs),
                chunks=len(chunks),
                index_dir=str(self.index_dir),
            )

    def _format_history_for_prompt(self) -> str:
        """Return a compact text of the last N turns for follow-up understanding."""
        n = max(0, int(self._history_prompt_turns))
        if n == 0:
            return ""

        with self._lock:
            items = list(self._history)[-n:]  # chronological (old -> recent)

        if not items:
            return ""

        lines: List[str] = []
        for i, it in enumerate(items, 1):
            q = it.get("question", "")
            a = it.get("answer", "")
            a_short = str(a).strip().replace("\n", " ")
            if len(a_short) > 400:
                a_short = a_short[:400] + "…"
            lines.append(f"Tour {i} - Q: {q}\nTour {i} - R: {a_short}")

        return "\n\n".join(lines)

    def _log_history(self, item: Dict[str, Any]) -> None:
        with self._lock:
            self._history.append(item)

    def get_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(reversed(self._history))  # most recent first

    def rate(self, turn_id: str, vote: int) -> bool:
        """vote = 1 (👍) or -1 (👎). Returns False if turn_id not found (too old / restarted)."""
        if vote not in (-1, 1):
            raise ValueError("vote must be -1 or 1")

        with self._lock:
            for it in self._history:
                if it.get("turn_id") == turn_id:
                    it["rating"] = vote
                    return True
        return False
