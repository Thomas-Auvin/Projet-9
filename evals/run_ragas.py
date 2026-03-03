from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import time

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from rag.moteur.engine import RagEngine
from rag.preparation.embeddings import get_mistral_embeddings


def find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Impossible de trouver la racine (pyproject.toml).")


ROOT = find_root()


def resolve_from_root(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


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


def load_golden_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """
    Map question -> golden record.
    Expected JSONL line example:
    {"question":"...", "expected_source_ids":["123","456"], "min_expected_matches":1, "k_eval":6}
    """
    m: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("question") or obj.get("user_input") or "").strip()
            if q:
                m[q] = obj
    return m


def compute_golden_metrics(
    expected_ids: list[Any],
    predicted_ids: list[Any],
) -> dict[str, float]:
    exp = [str(x) for x in (expected_ids or [])]
    pred = [str(x) for x in (predicted_ids or [])]

    if not exp:
        return {
            "gold_n_expected": float("nan"),
            "gold_n_hit": float("nan"),
            "gold_hit_at_k": float("nan"),
            "gold_precision_at_k": float("nan"),
            "gold_recall_at_k": float("nan"),
            "gold_mrr": float("nan"),
        }

    exp_set = set(exp)
    n_hit = sum(1 for pid in pred if pid in exp_set)

    hit_at_k = 1.0 if n_hit > 0 else 0.0
    precision_at_k = n_hit / max(len(pred), 1)
    recall_at_k = n_hit / len(exp)

    mrr = 0.0
    for i, pid in enumerate(pred, start=1):
        if pid in exp_set:
            mrr = 1.0 / i
            break

    return {
        "gold_n_expected": float(len(exp)),
        "gold_n_hit": float(n_hit),
        "gold_hit_at_k": float(hit_at_k),
        "gold_precision_at_k": float(precision_at_k),
        "gold_recall_at_k": float(recall_at_k),
        "gold_mrr": float(mrr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=str,
        default="evals/datasets/questions.jsonl",
        help="JSONL with {'question' or 'user_input'}",
    )
    parser.add_argument(
        "--golden",
        type=str,
        default="",
        help="Optional JSONL with expected_source_ids (can be same as --questions).",
    )
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--outdir", type=str, default="evals/experiments")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep (seconds) between questions to avoid rate limits.",
    )
    args = parser.parse_args()

    load_dotenv()

    questions_path = resolve_from_root(args.questions)
    outdir = resolve_from_root(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    limit = None if args.limit == 0 else args.limit
    questions = load_questions_jsonl(questions_path, limit=limit)
    if not questions:
        raise SystemExit(f"No questions found in {questions_path}")

    golden_map: dict[str, dict[str, Any]] = {}
    golden_path: Path | None = None
    if args.golden:
        golden_path = resolve_from_root(args.golden)
        golden_map = load_golden_jsonl(golden_path)
        if not golden_map:
            raise SystemExit(f"No golden records found in {golden_path}")

    print("Loaded questions:", len(questions))
    print("First 3:", questions[:3])
    if golden_path:
        covered = sum(1 for q in questions if q in golden_map)
        print(f"Golden enabled: {golden_path} | coverage: {covered}/{len(questions)}")

    # --- Prompt controls (driven by env) ---
    prompt_variant_env = os.environ.get("P9_PROMPT_VARIANT", "v2")
    prompt_version_env = os.environ.get("P9_PROMPT_VERSION", "v2_1")
    debug_prompt = os.environ.get("P9_DEBUG_PROMPT", "0")
    print(
        f"Prompt: variant={prompt_variant_env} version={prompt_version_env} debug={debug_prompt}"
    )

    # --- Init your RAG engine (same config as the API) ---
    index_dir = resolve_from_root(
        os.environ.get("P9_INDEX_DIR", "artifacts/faiss_index_mistral")
    )
    processed_csv = resolve_from_root(
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

    rows: list[dict[str, Any]] = []
    for q in questions:
        gold = golden_map.get(q) if golden_map else None
        k_for_q = int(gold.get("k_eval", args.k)) if gold else int(args.k)
        min_expected_matches = (
            int(gold.get("min_expected_matches", 1)) if gold else None
        )

        res = engine.ask(q, k=k_for_q)

        # IMPORTANT: requires the tiny patch in engine.ask()
        ctxs = res.get("retrieved_contexts", [])
        if not isinstance(ctxs, list):
            ctxs = []

        # Réduire la charge du juge (très utile)
        max_ctx = int(os.getenv("P9_RAGAS_MAX_CONTEXTS", "4"))
        max_chars = int(os.getenv("P9_RAGAS_MAX_CONTEXT_CHARS", "1200"))
        ctxs = [str(c)[:max_chars] for c in ctxs[:max_ctx]]

        ans = str(res.get("answer", ""))
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

        # --- Golden metrics (no extra model call) ---
        sources = res.get("sources", []) or []
        pred_ids: list[str] = []
        if isinstance(sources, list):
            for s in sources:
                if isinstance(s, dict) and s.get("id") is not None:
                    pred_ids.append(str(s["id"]))

        if gold:
            expected_ids = gold.get("expected_source_ids") or []
            gold_metrics = compute_golden_metrics(expected_ids, pred_ids)
            n_hit = int(gold_metrics["gold_n_hit"])
            gold_metrics["gold_pass_min_matches"] = (
                1.0 if n_hit >= min_expected_matches else 0.0
            )
            golden_used = True
        else:
            gold_metrics = {
                "gold_n_expected": float("nan"),
                "gold_n_hit": float("nan"),
                "gold_hit_at_k": float("nan"),
                "gold_precision_at_k": float("nan"),
                "gold_recall_at_k": float("nan"),
                "gold_mrr": float("nan"),
                "gold_pass_min_matches": float("nan"),
            }
            golden_used = False

        rows.append(
            {
                "question": q,
                "answer": ans,
                "k": k_for_q,
                "faithfulness": f,
                "answer_relevancy": ar,
                "n_contexts": len(ctxs),
                "turn_id": res.get("turn_id"),
                # Prompt tracking
                "prompt_variant": res.get("prompt_variant", prompt_variant_env),
                "prompt_version": res.get("prompt_version", prompt_version_env),
                # Meta
                "chat_model": res.get("model", chat_model),
                "judge_model": judge_model,
                "index_dir": str(index_dir),
                "processed_csv": str(processed_csv),
                # Golden
                "golden_used": golden_used,
                "gold_min_expected_matches": min_expected_matches,
                **gold_metrics,
            }
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_variant = (prompt_variant_env or "v2").replace("/", "-")
    out_csv = outdir / f"ragas_eval_{safe_variant}_{stamp}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print("Saved:", out_csv)
    print(df[["faithfulness", "answer_relevancy"]].mean(numeric_only=True))

    if "golden_used" in df.columns and df["golden_used"].any():
        gold_cols = [
            "gold_hit_at_k",
            "gold_precision_at_k",
            "gold_recall_at_k",
            "gold_mrr",
            "gold_pass_min_matches",
        ]
        print(df.loc[df["golden_used"], gold_cols].mean(numeric_only=True))


if __name__ == "__main__":
    main()
