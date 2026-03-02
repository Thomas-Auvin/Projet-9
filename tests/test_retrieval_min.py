from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from rag.preparation.embeddings import DeterministicEmbeddings
from rag.preparation.indexing import (
    SplitConfig,
    build_faiss_index,
    load_faiss_index,
    split_documents,
)


def test_retrieval_min_identical_doc_is_top1(tmp_path: Path) -> None:
    """
    Minimal retrieval evaluation (CI-friendly):
    - tiny fixed corpus
    - force 1 chunk per document (very large chunk_size, no overlap)
    - query with text identical to the target doc
    - expect the target doc to be ranked #1
    """

    docs = [
        Document(
            page_content=(
                "Titre: Expo peinture\n"
                "Ville: Bordeaux\n"
                "Description: Une exposition d'art moderne."
            ),
            metadata={"event_id": "1", "title": "Expo peinture", "city": "Bordeaux"},
        ),
        Document(
            page_content=(
                "Titre: Concert jazz\n"
                "Ville: Bayonne\n"
                "Description: Concert jazz en plein air."
            ),
            metadata={"event_id": "2", "title": "Concert jazz", "city": "Bayonne"},
        ),
        Document(
            page_content=(
                "Titre: Match basket\n"
                "Ville: Toulouse\n"
                "Description: Match de basket samedi soir, entrée 5 euros."
            ),
            metadata={"event_id": "3", "title": "Match basket", "city": "Toulouse"},
        ),
    ]

    # Force 1 chunk per doc to avoid duplicates (multiple chunks with same title).
    chunks = split_documents(docs, SplitConfig(chunk_size=10_000, chunk_overlap=0))
    assert len(chunks) == len(docs)

    emb = DeterministicEmbeddings(dim=64)

    index_dir = tmp_path / "faiss_index"
    build_faiss_index(chunks, emb, index_dir)

    vs = load_faiss_index(index_dir, emb)

    # Query identical to doc #2 content => should be nearest neighbor.
    query = docs[1].page_content
    res = vs.similarity_search(query, k=3)

    assert len(res) == 3

    top = res[0]
    assert top.metadata.get("event_id") == "2"
    assert top.metadata.get("title") == "Concert jazz"
