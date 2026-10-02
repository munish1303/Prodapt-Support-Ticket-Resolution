"""Retrieval evaluation against TREC-style graded qrels.

Relevance grades (data/evaluation/retrieval_qrels.tsv, built by construction):
  2 = resolved ticket or KB article for the same root-cause scenario as the query
  1 = resolved ticket / KB article with the same intent and an overlapping product
      (related, partially useful)
Metrics: nDCG@10 (graded), MRR@20 / P@k / Hit@k / capped Recall@k on grade-2 docs,
and KB-Recall@k (is the scenario's KB article retrieved?).
"""

from __future__ import annotations

import time

import numpy as np

from evaluation.metrics import (
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
