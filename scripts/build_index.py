from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from project_paths import ARTIFACTS_DIR, PROCESSED_DIR
from rag.embeddings import DeterministicEmbeddings, get_mistral_embeddings
from rag.indexing import (
    SplitConfig,
    build_faiss_index,
    events_to_documents,
    load_processed_events,
    split_documents,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build FAISS index from processed events CSV."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "events_processed_geo.csv",
        help="Path to processed CSV (sep=';').",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ARTIFACTS_DIR / "faiss_index",
        help="Directory where FAISS index will be saved.",
    )
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--chunk-overlap", type=int, default=120)
    p.add_argument(
        "--use-dummy-embeddings",
        action="store_true",
        help="Use deterministic embeddings (no API key) for dry-run/tests.",
    )
    p.add_argument(
        "--mistral-embed-model",
        type=str,
        default="mistral-embed",
        help="Mistral embedding model name.",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()
    df = load_processed_events(args.input)

    docs = events_to_documents(df)
    cfg = SplitConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = split_documents(docs, cfg)

    if args.use_dummy_embeddings:
        embeddings = DeterministicEmbeddings(dim=64)
        print("Embeddings: DeterministicEmbeddings(dim=64)")
    else:
        embeddings = get_mistral_embeddings(model=args.mistral_embed_model)
        print(f"Embeddings: MistralAIEmbeddings(model={args.mistral_embed_model})")

    print(f"Loaded rows: {len(df)}")
    print(f"Documents:   {len(docs)}")
    print(f"Chunks:      {len(chunks)}")

    build_faiss_index(chunks, embeddings, args.out)
    print(f"Saved FAISS index to: {args.out}")


if __name__ == "__main__":
    main()
