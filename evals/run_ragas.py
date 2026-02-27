from __future__ import annotations
from pathlib import Path

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from rag.embeddings import get_mistral_embeddings
from rag.engine import RagEngine

from ragas.metrics.collections import AnswerRelevancy, Faithfulness
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
import math


def find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Impossible de trouver la racine (pyproject.toml).")


ROOT = find_root()


def load_questions_jsonl(path: Path, limit: int | None = None) -> list[str]:
    qs: list[str] = []
    # utf-8-sig gère aussi un éventuel BOM Windows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("user_input") or obj.get("question") or "").strip()
            if q:
                qs.append(q)
            if limit is not None and len(qs) >= limit:
                break
    return qs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions", type=str, default="evals/datasets/questions.jsonl"
    )
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--outdir", type=str, default="evals/experiments")
    args = parser.parse_args()

    load_dotenv()

    questions_path = Path(args.questions)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    limit = None if args.limit == 0 else args.limit
    questions = load_questions_jsonl(questions_path, limit=limit)
    if not questions:
        raise SystemExit(f"No questions found in {questions_path}")

    print("Loaded questions:", len(questions))
    print("First 3:", questions[:3])

    # --- Init your RAG engine (same config as the API) ---
    index_dir = Path(os.environ.get("P9_INDEX_DIR", "artifacts/faiss_index_mistral"))
    processed_csv = Path(
        os.environ.get("P9_PROCESSED_CSV", "data/processed/events_processed_geo.csv")
    )
    k_default = int(os.environ.get("P9_K_DEFAULT", "6"))
    chat_model = os.environ.get("MISTRAL_CHAT_MODEL", "mistral-small-latest")
    embed_model = os.environ.get("MISTRAL_EMBED_MODEL", "mistral-embed")

    embeddings = get_mistral_embeddings(model=embed_model)

    engine = RagEngine(
        index_dir=index_dir,
        processed_csv=processed_csv,
        embeddings=embeddings,
        chat_model=chat_model,
        k_default=k_default,
    )
    engine.load()

    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise SystemExit("MISTRAL_API_KEY manquant (nécessaire pour RAGAS).")

    base_url = os.getenv("P9_MISTRAL_OPENAI_BASE_URL", "https://api.mistral.ai/v1")

    # Client OpenAI configuré sur l'API Mistral (OpenAI-compatible)
    client = AsyncOpenAI(api_key=mistral_key, base_url=base_url)

    judge_model = os.getenv("P9_RAGAS_JUDGE_MODEL", "mistral-small-latest")
    embed_model = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")

    # LLM "juge" moderne (InstructorLLM) requis par les metrics collections
    judge_max_tokens = int(os.getenv("P9_RAGAS_MAX_TOKENS", "4096"))

    judge_llm = llm_factory(
        judge_model,
        provider="openai",
        client=client,
        temperature=0,
        top_p=1.0,
        max_tokens=judge_max_tokens,
        system_prompt=(
            "You are an evaluator of RAG systems. "
            "Return only valid JSON for structured outputs. "
            "Keep reasons extremely concise (<=10 words)."
        ),
    )

    # Embeddings modernes (API /v1/embeddings côté Mistral)
    judge_emb = OpenAIEmbeddings(client=client, model=embed_model)

    faithfulness = Faithfulness(llm=judge_llm)
    answer_rel = AnswerRelevancy(llm=judge_llm, embeddings=judge_emb)

    rows: list[dict] = []
    for q in questions:
        res = engine.ask(q, k=args.k)

        # IMPORTANT: requires the tiny patch in engine.ask()
        ctxs = res.get("retrieved_contexts", [])
        if not isinstance(ctxs, list):
            ctxs = []
        # Réduire la charge du juge (très utile)
        max_ctx = int(os.getenv("P9_RAGAS_MAX_CONTEXTS", "4"))
        max_chars = int(os.getenv("P9_RAGAS_MAX_CONTEXT_CHARS", "1200"))
        ctxs = [c[:max_chars] for c in ctxs[:max_ctx]]
        ans = res.get("answer", "")
        max_ans_chars = int(os.getenv("P9_RAGAS_MAX_ANSWER_CHARS", "1200"))
        ans = ans[:max_ans_chars]

        try:
            f = faithfulness.score(
                user_input=q, response=ans, retrieved_contexts=ctxs
            ).value
        except Exception as e:
            print("Faithfulness failed for:", q, "|", type(e).__name__, str(e)[:120])
            f = math.nan

        try:
            ar = answer_rel.score(user_input=q, response=ans).value
        except Exception as e:
            print("AnswerRelevancy failed for:", q, "|", type(e).__name__, str(e)[:120])
            ar = math.nan

        rows.append(
            {
                "question": q,
                "answer": ans,
                "k": args.k,
                "faithfulness": f,
                "answer_relevancy": ar,
                "n_contexts": len(ctxs),
                "turn_id": res.get("turn_id"),
            }
        )

    df = pd.DataFrame(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = outdir / f"ragas_eval_{stamp}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print("Saved:", out_csv)
    print(df[["faithfulness", "answer_relevancy"]].mean(numeric_only=True))


if __name__ == "__main__":
    main()
