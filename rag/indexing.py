from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


REQUIRED_COLS = [
    "event_id",
    "title",
    "url",
    "doc_text",
    "event_start",
    "event_end",
    "city",
    "department",
    "region",
    "lat",
    "lon",
    "is_upcoming",
]


def load_processed_events(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", low_memory=False)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in processed CSV: {missing}")

    for c in ["event_start", "event_end"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    df["event_id"] = df["event_id"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["url"] = df["url"].fillna("").astype(str)
    df["doc_text"] = df["doc_text"].fillna("").astype(str)

    def _parse_bool(x) -> bool:
        if x is None or pd.isna(x):
            return False
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(int(x))
        s = str(x).strip().lower()
        return s in {"1", "true", "t", "yes", "y", "vrai"}

    df["is_upcoming"] = df["is_upcoming"].map(_parse_bool)

    return df


def _iso_or_empty(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    try:
        # pandas Timestamp
        return pd.Timestamp(x).isoformat()
    except Exception:
        return str(x)


def events_to_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert rows to LangChain Documents.
    doc_text -> page_content
    Other useful fields -> metadata
    """
    docs: list[Document] = []
    for _, row in df.iterrows():
        meta = {
            "event_id": row["event_id"],
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "city": row.get("city", ""),
            "department": row.get("department", ""),
            "region": row.get("region", ""),
            "event_start": _iso_or_empty(row.get("event_start")),
            "event_end": _iso_or_empty(row.get("event_end")),
            "lat": row.get("lat"),
            "lon": row.get("lon"),
            "is_upcoming": bool(row.get("is_upcoming", False)),
        }
        docs.append(Document(page_content=row["doc_text"], metadata=meta))
    return docs


@dataclass(frozen=True)
class SplitConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120


def split_documents(docs: list[Document], cfg: SplitConfig = SplitConfig()) -> list[Document]:
    """
    Split documents into chunks for embedding/retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_faiss_index(
    chunks: list[Document],
    embeddings: Any,
    index_dir: Path,
) -> FAISS:
    """
    Build + save FAISS index.
    embeddings must implement embed_documents/embed_query (LangChain Embeddings interface).
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(index_dir))
    return vs


def load_faiss_index(index_dir: Path, embeddings: Any) -> FAISS:
    """
    Load FAISS index saved with save_local.
    """
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
