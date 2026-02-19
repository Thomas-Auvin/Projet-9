from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from project_paths import ARTIFACTS_DIR
from rag.embeddings import DeterministicEmbeddings, get_mistral_embeddings
from rag.indexing import load_faiss_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query FAISS index.")
    p.add_argument("--index-dir", type=Path, default=ARTIFACTS_DIR / "faiss_index")
    p.add_argument("--q", type=str, required=True, help="Question / query text.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--use-dummy-embeddings", action="store_true")
    p.add_argument("--mistral-embed-model", type=str, default="mistral-embed")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.use_dummy_embeddings:
        embeddings = DeterministicEmbeddings(dim=64)
    else:
        embeddings = get_mistral_embeddings(model=args.mistral_embed_model)

    vs = load_faiss_index(args.index_dir, embeddings)
    docs = vs.similarity_search(args.q, k=args.k)

    print(f"Query: {args.q}")
    print(f"Top-{args.k} results:\n")
    for i, d in enumerate(docs, 1):
        md = d.metadata
        print(f"[{i}] {md.get('title','')}")
        print(f"    url: {md.get('url','')}")
        print(f"    city: {md.get('city','')} | dept: {md.get('department','')} | region: {md.get('region','')}")
        print(f"    start: {md.get('event_start','')} | end: {md.get('event_end','')}")
        print(f"    upcoming: {md.get('is_upcoming', False)}")
        print(f"    snippet: {d.page_content[:180].replace('\\n',' ')}")
        print("")


if __name__ == "__main__":
    main()
