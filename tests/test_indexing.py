from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from rag.embeddings import DeterministicEmbeddings
from rag.indexing import (
    SplitConfig,
    build_faiss_index,
    load_faiss_index,
    split_documents,
)


def test_build_and_query_faiss(tmp_path: Path) -> None:
    docs = [
        Document(
            page_content="Titre: Expo peinture\nVille: Bordeaux\nDescription: Une exposition d'art moderne.",
            metadata={"event_id": "1", "title": "Expo peinture", "city": "Bordeaux"},
        ),
        Document(
            page_content="Titre: Concert jazz\nVille: Bayonne\nDescription: Concert jazz en plein air.",
            metadata={"event_id": "2", "title": "Concert jazz", "city": "Bayonne"},
        ),
    ]

    chunks = split_documents(docs, SplitConfig(chunk_size=80, chunk_overlap=10))
    emb = DeterministicEmbeddings(dim=64)

    index_dir = tmp_path / "faiss_index"
    build_faiss_index(chunks, emb, index_dir)

    vs = load_faiss_index(index_dir, emb)
    res = vs.similarity_search("jazz", k=2)

    assert len(res) >= 1
    titles = [d.metadata.get("title") for d in res]
    assert "Concert jazz" in titles
