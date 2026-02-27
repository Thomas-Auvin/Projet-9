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
    HistoryResponse,
    HistoryItem,
    FeedbackRequest,
    FeedbackResponse,
)
from rag.embeddings import get_mistral_embeddings
from rag.engine import RagEngine

ENGINE: Optional[RagEngine] = None


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


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


@app.get("/history", response_model=HistoryResponse)
def history(engine: RagEngine = Depends(get_engine)) -> HistoryResponse:
    items = engine.get_history()
    return HistoryResponse(items=[HistoryItem(**it) for it in items])


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
