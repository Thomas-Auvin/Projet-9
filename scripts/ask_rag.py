from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from project_paths import ARTIFACTS_DIR
from rag.embeddings import get_mistral_embeddings
from rag.rag_chain import answer_question


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ask the RAG system (FAISS + Mistral).")
    p.add_argument(
        "--index-dir", type=Path, default=ARTIFACTS_DIR / "faiss_index_mistral"
    )
    p.add_argument("--q", type=str, required=True)
    p.add_argument("--k", type=int, default=6)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    embeddings = get_mistral_embeddings(model="mistral-embed")
    res = answer_question(
        question=args.q,
        index_dir=str(args.index_dir),
        embeddings=embeddings,
        k=args.k,
    )

    print(res["answer"])


if __name__ == "__main__":
    main()
