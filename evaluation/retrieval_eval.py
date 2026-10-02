"""Retrieval evaluation against TREC-style graded qrels.

Relevance grades (data/evaluation/retrieval_qrels.tsv, built by construction):
  2 = resolved ticket or KB article for the same root-cause scenario as the query
  1 = resolved ticket / KB article with the same intent and an overlapping product
      (related, partially useful)
Metrics: nDCG@10 (graded), MRR@20 / P@k / Hit@k / capped Recall@k on grade-2 docs,
and KB-Recall@k (is the scenario's KB article retrieved?).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python evaluation/retrieval_eval.py`

from evaluation.metrics import (  # noqa: E402
    capped_recall_at_k,
    hit_at_k,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    reciprocal_rank,
)


def per_query_metrics(ranked: list[str], grades: dict[str, int]) -> dict[str, float]:
    highly = {d for d, g in grades.items() if g >= 2}
    kb = {d for d in highly if d.startswith("KB-")}
    return {
        "ndcg@10": ndcg_at_k(ranked, grades, 10),
        "mrr": reciprocal_rank(ranked[:20], highly),
        "p@5": precision_at_k(ranked, highly, 5),
        "p@10": precision_at_k(ranked, highly, 10),
        "hit@1": hit_at_k(ranked, highly, 1),
        "hit@5": hit_at_k(ranked, highly, 5),
        "recall@10_capped": capped_recall_at_k(ranked, highly, 10),
        "recall@20_capped": capped_recall_at_k(ranked, highly, 20),
        "kb_recall@5": hit_at_k(ranked, kb, 5) if kb else 0.0,
        "kb_recall@10": hit_at_k(ranked, kb, 10) if kb else 0.0,
    }


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    keys = per_query[0].keys() if per_query else []
    return {k: round(mean([q[k] for q in per_query]), 4) for k in keys}


async def run_queries(search_fn, queries: list[dict], qrels: dict[str, dict[str, int]]) -> dict:
    """search_fn(query_text) -> ranked list of doc ids (awaitable)."""
    per_query, latencies = [], []
    for q in queries:
        t0 = time.perf_counter()
        ranked = await search_fn(q["complaint"])
        latencies.append((time.perf_counter() - t0) * 1000)
        per_query.append(per_query_metrics(ranked, qrels.get(q["id"], {})))
    return {
        "metrics": aggregate(per_query),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "mean": round(float(np.mean(latencies)), 1),
        },
        "per_query": per_query,
    }


async def _main() -> None:
    """Evaluate the *production* retrieval configuration (hybrid RRF with the configured weights, KB slots, context size and
    reranking setting) on the 200 held-out queries. E1/E2 compare methods; this reports what is deployed."""
    import json

    from app.config import settings
    from app.core.database import dispose_engine, get_session_factory
    from app.models.embeddings import get_embedding_service
    from app.services.retrieval import HybridRetriever, Reranker, RetrievalService
    from evaluation.data import load_eval, load_qrels, save_result

    sf = get_session_factory()
    service = RetrievalService(
        HybridRetriever(sf, get_embedding_service()), Reranker() if settings.USE_RERANKING else None
    )

    async def search(q: str) -> list[str]:
        res = await service.retrieve(q, top_k=settings.RETRIEVAL_TOP_K)  # what the generator actually sees
        return [i.document.id for i in res.items]

    queries, qrels = load_eval("retrieval_eval"), load_qrels()
    await search(queries[0]["complaint"])  # warm-up
    out = await run_queries(search, queries, qrels)
    out.pop("per_query")
    out["config"] = {
        "rrf_weights": [settings.RRF_SEMANTIC_WEIGHT, settings.RRF_LEXICAL_WEIGHT],
        "kb_min_slots": settings.RETRIEVAL_KB_MIN_SLOTS,
        "lexical_query_mode": settings.LEXICAL_QUERY_MODE,
        "use_reranking": settings.USE_RERANKING,
        "top_k": settings.RETRIEVAL_TOP_K,
    }
    print(json.dumps(out, indent=2))
    print(f"saved {save_result('retrieval_production', out)}")
    await dispose_engine()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
