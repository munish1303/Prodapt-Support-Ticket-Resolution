"""Experiment 2 - Cross-encoder reranking (MUST RUN).

Hypothesis: reranking hybrid top-30 with cross-encoder/ms-marco-MiniLM-L-6-v2
improves Precision@5 by > 0.05 (absolute) with < 2 s added latency.
Decision rule (plan §34.2): enable USE_RERANKING only if both hold.

Usage: python experiments/e2_reranking.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.database import dispose_engine, get_session_factory  # noqa: E402
from app.models.embeddings import get_embedding_service  # noqa: E402
from app.services.retrieval import HybridRetriever, Reranker  # noqa: E402
from evaluation.data import load_eval, load_qrels, save_result  # noqa: E402
from evaluation.metrics import paired_bootstrap_p, percentile  # noqa: E402
from evaluation.retrieval_eval import aggregate, per_query_metrics  # noqa: E402

CANDIDATES, FINAL_K = 30, 10
MIN_P5_GAIN, MAX_ADDED_LATENCY_MS = 0.05, 2000


async def main() -> None:
    sf = get_session_factory()
    retriever = HybridRetriever(sf, get_embedding_service())
    reranker = Reranker()
    queries, qrels = load_eval("retrieval_eval"), load_qrels()

    base_pq, rr_pq, base_lat, rr_lat = [], [], [], []
    await retriever.retrieve(queries[0]["complaint"], top_k=CANDIDATES, kb_min_slots=0)
    reranker.rerank("warm up", [], 1)
    for q in queries:
        t0 = time.perf_counter()
        base = await retriever.retrieve(q["complaint"], top_k=FINAL_K, kb_min_slots=0)
        base_lat.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        cands = await retriever.retrieve(q["complaint"], top_k=CANDIDATES, kb_min_slots=0)
        reranked = reranker.rerank(q["complaint"], cands.items, FINAL_K)
        rr_lat.append((time.perf_counter() - t0) * 1000)

        grades = qrels.get(q["id"], {})
        base_pq.append(per_query_metrics([i.document.id for i in base.items], grades))
        rr_pq.append(per_query_metrics([i.document.id for i in reranked], grades))

    b, r = aggregate(base_pq), aggregate(rr_pq)
    p5_gain = r["p@5"] - b["p@5"]
    added = float(np.mean(rr_lat) - np.mean(base_lat))
    decision = (
        "ENABLE reranking" if p5_gain > MIN_P5_GAIN and added < MAX_ADDED_LATENCY_MS else "KEEP reranking disabled"
    )
    payload = {
        "n_queries": len(queries),
        "candidates": CANDIDATES,
        "final_k": FINAL_K,
        "hybrid": b,
        "hybrid_reranked": r,
        "p@5_abs_gain": round(p5_gain, 4),
        "p@5_rel_gain": round(p5_gain / b["p@5"], 4) if b["p@5"] else None,
        "ndcg@10_abs_gain": round(r["ndcg@10"] - b["ndcg@10"], 4),
        "p_value_p@5": paired_bootstrap_p([x["p@5"] for x in rr_pq], [x["p@5"] for x in base_pq]),
        "latency_ms": {
            "hybrid_mean": round(float(np.mean(base_lat)), 1),
            "hybrid_p95": round(percentile(base_lat, 95), 1),
            "reranked_mean": round(float(np.mean(rr_lat)), 1),
            "reranked_p95": round(percentile(rr_lat, 95), 1),
            "added_mean": round(added, 1),
        },
        "decision_rule": f"enable iff P@5 gain > {MIN_P5_GAIN} and added latency < {MAX_ADDED_LATENCY_MS} ms",
        "decision": decision,
    }
    print(
        f"P@5 {b['p@5']:.3f} -> {r['p@5']:.3f} (gain {p5_gain:+.3f}); nDCG@10 {b['ndcg@10']:.3f} -> {r['ndcg@10']:.3f}; "
        f"added latency {added:.0f} ms => {decision}"
    )
    print(f"saved {save_result('e2_reranking', payload)}")
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
