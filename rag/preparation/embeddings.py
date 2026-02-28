from __future__ import annotations

import hashlib
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings

try:
    # Real embeddings (V1)
    from langchain_mistralai.embeddings import MistralAIEmbeddings
except Exception:  # pragma: no cover
    MistralAIEmbeddings = None  # type: ignore


class DeterministicEmbeddings(Embeddings):
    """
    Deterministic embeddings for unit tests / dry-run (no network, no API key).
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        vec = np.resize(arr, self.dim) / 255.0
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_mistral_embeddings(model: str = "mistral-embed") -> Embeddings:
    """
    V1: Mistral-only embeddings.
    Requires env var MISTRAL_API_KEY to be set.
    """
    if MistralAIEmbeddings is None:
        raise RuntimeError("langchain-mistralai not available. Did you install it?")

    # MistralAIEmbeddings reads MISTRAL_API_KEY from env
    return MistralAIEmbeddings(model=model)
