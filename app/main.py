from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException

from app.schemas import (
    AskRequest,
    AskResponse,
    RebuildResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from rag.preparation.embeddings import get_mistral_embeddings
from rag.moteur.engine import RagEngine
import json
from datetime import datetime, timezone

from project_paths import OUTPUTS_DIR

ENGINE: Optional[RagEngine] = None


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def _feedback_path() -> Path:
    # Override test/docker friendly. Default: outputs/feedback.jsonl
    return Path(os.environ.get("P9_FEEDBACK_PATH", str(OUTPUTS_DIR / "feedback.jsonl")))


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _find_turn_in_history(engine: RagEngine, turn_id: str) -> dict | None:
    # engine.get_history() renvoie une liste de dicts (cf /history) :contentReference[oaicite:4]{index=4}
    try:
        items = engine.get_history()
    except Exception:
        return None
    for it in items or []:
        if it.get("turn_id") == turn_id:
            return it
    return None


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_engine() -> RagEngine:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized.")
    return ENGINE


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    load_dotenv()

    index_dir = _env_path("P9_INDEX_DIR", "artifacts/faiss_index_mistral")
    processed_csv = _env_path(
        "P9_PROCESSED_CSV", "data/processed/events_processed_geo.csv"
    )
    k_default = int(os.environ.get("P9_K_DEFAULT", "6"))
    chat_model = os.environ.get("MISTRAL_CHAT_MODEL", "mistral-small-latest")
    embed_model = os.environ.get("MISTRAL_EMBED_MODEL", "mistral-embed")

    embeddings = get_mistral_embeddings(model=embed_model)

    history_size = int(os.getenv("P9_HISTORY_SIZE", "10"))
    history_prompt_turns = int(os.getenv("P9_HISTORY_PROMPT_TURNS", "3"))

    ENGINE = RagEngine(
        index_dir=index_dir,
        processed_csv=processed_csv,
        embeddings=embeddings,
        chat_model=chat_model,
        k_default=k_default,
        history_size=history_size,
        history_prompt_turns=history_prompt_turns,
    )

    # On tente de charger l'index s'il existe déjà.
    try:
        ENGINE.load()
    except Exception:
        # Pas bloquant : /rebuild pourra le créer.
        pass

    yield

    ENGINE = None


app = FastAPI(title="Puls-Events RAG POC", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, engine: RagEngine = Depends(get_engine)) -> AskResponse:
    try:
        res = engine.ask(req.question, k=req.k)
        return AskResponse(**res)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(
    req: FeedbackRequest, engine: RagEngine = Depends(get_engine)
) -> FeedbackResponse:
    try:
        ok = engine.rate(req.turn_id, req.vote)
        if not ok:
            raise HTTPException(
                status_code=404,
                detail="turn_id not found (too old or server restarted).",
            )

        # Build record (minimal + enrichi si on retrouve le turn dans l'historique)
        record: dict = {
            "ts_utc": _iso_utc_now(),
            "turn_id": req.turn_id,
            "vote": req.vote,
        }

        ctx = _find_turn_in_history(engine, req.turn_id)
        if ctx:
            # On logge les champs utiles s'ils existent.
            for k in [
                "question",
                "answer",
                "sources",
                "model",
                "k",
                "rating",
                "prompt_variant",
                "prompt_version",
            ]:
                if k in ctx:
                    record[k] = ctx.get(k)

        try:
            _append_jsonl(_feedback_path(), record)
        except OSError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to persist feedback: {e}"
            ) from e

        return FeedbackResponse(status="ok", turn_id=req.turn_id, rating=req.vote)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/rebuild", response_model=RebuildResponse)
def rebuild(
    x_rebuild_token: str | None = Header(default=None, alias="X-Rebuild-Token"),
    engine: RagEngine = Depends(get_engine),
) -> RebuildResponse:
    token = os.environ.get("REBUILD_TOKEN", "")
    if not token:
        raise HTTPException(
            status_code=403, detail="REBUILD_TOKEN not set; rebuild disabled."
        )
    if x_rebuild_token != token:
        raise HTTPException(status_code=403, detail="Invalid rebuild token.")

    try:
        stats = engine.rebuild()
        return RebuildResponse(
            status="ok",
            rows=stats.rows,
            docs=stats.docs,
            chunks=stats.chunks,
            index_dir=stats.index_dir,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
