from __future__ import annotations

import os
from typing import Any
import random
import time

from dotenv import load_dotenv
from mistralai import Mistral
from langchain_core.documents import Document

from rag.preparation.indexing import load_faiss_index


def _short(s: str, n: int = 650) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s[:n] + ("…" if len(s) > n else "")


def build_context(docs: list[Document]) -> tuple[str, list[dict[str, str]]]:
    """
    Returns:
      - context_text: a compact context block for the LLM
      - sources: list of {id,title,url,city,start,end}
    """
    blocks: list[str] = []
    sources: list[dict[str, str]] = []

    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        raw_id = md.get("event_id")
        sid = str(raw_id) if raw_id not in (None, "") else f"S{i}"
        title = str(md.get("title", "") or "")
        url = str(md.get("url", "") or "")
        city = str(md.get("city", "") or "")
        start = str(md.get("event_start", "") or "")
        end = str(md.get("event_end", "") or "")

        sources.append(
            {
                "id": sid,
                "title": title,
                "url": url,
                "city": city,
                "start": start,
                "end": end,
            }
        )

        blocks.append(
            f"[{sid}] {title}\n"
            f"Ville: {city}\n"
            f"Dates: {start} → {end}\n"
            f"URL: {url}\n"
            f"Extrait: {_short(d.page_content)}"
        )

    return "\n\n".join(blocks), sources


def _mistral_chat(client: Mistral, model: str, system: str, user: str) -> str:
    """
    Compatible wrapper for the Mistral Python client.
    Returns the generated text content (str).
    Includes retry/backoff on HTTP 429 (rate limit).
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    max_retries = int(os.getenv("P9_MISTRAL_MAX_RETRIES", "6"))
    base_sleep = float(os.getenv("P9_MISTRAL_RETRY_BASE_SLEEP", "1.5"))

    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.complete(model=model, messages=messages, temperature=0.2)

            # Try common SDK shapes to extract the text.
            # Shape A (OpenAI-like): resp.choices[0].message.content
            try:
                content = resp.choices[0].message.content  # type: ignore[attr-defined]
                if content is not None:
                    return str(content)
            except Exception:
                pass

            # Shape B (newer structured output): resp.output[0].content[0].text
            try:
                content = resp.output[0].content[0].text  # type: ignore[attr-defined]
                if content is not None:
                    return str(content)
            except Exception:
                pass

            # Fallback: stringify the response (last resort)
            return str(resp)

        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            is_429 = (
                ("status 429" in msg)
                or ("rate limit" in msg)
                or ("rate_limited" in msg)
            )
            if not is_429 or attempt == max_retries - 1:
                raise

            # exponential backoff + jitter
            sleep_s = base_sleep * (2**attempt) + random.uniform(0.0, 0.5)
            time.sleep(sleep_s)

    # Normally unreachable
    raise RuntimeError("Mistral chat retry loop exhausted") from last_exc


def answer_question(
    question: str,
    index_dir: str,
    embeddings: Any,
    k: int = 6,
    chat_model: str | None = None,
) -> dict[str, Any]:
    load_dotenv()
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY manquant. Vérifie ton fichier .env.")

    model = chat_model or os.environ.get("MISTRAL_CHAT_MODEL", "mistral-small-latest")

    vs = load_faiss_index(index_dir=index_dir, embeddings=embeddings)
    docs = vs.similarity_search(question, k=k)

    context_text, sources = build_context(docs)

    system = (
        "Tu es un assistant qui recommande des événements culturels en Nouvelle-Aquitaine.\n"
        "Tu dois répondre UNIQUEMENT avec les informations du CONTEXTE fourni.\n"
        "Si le contexte ne permet pas de répondre, dis-le clairement et propose une reformulation.\n"
        "Réponds en français, de manière concise et utile.\n"
        "Donne au maximum 5 recommandations.\n"
        "Pour chaque recommandation : titre, date(s), ville, et URL.\n"
        "Termine par une section 'Sources' listant les IDs [S1..] utilisés."
    )

    user = f"QUESTION:\n{question}\n\nCONTEXTE:\n{context_text}\n\nRÉPONSE:"

    client = Mistral(api_key=api_key)
    answer = _mistral_chat(client, model=model, system=system, user=user)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "k": k,
        "model": model,
    }
