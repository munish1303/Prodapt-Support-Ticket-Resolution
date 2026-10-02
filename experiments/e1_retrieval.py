"""Experiment 1 - Retrieval method comparison (MUST RUN).

Hypothesis: hybrid (pgvector + PostgreSQL FTS, fused with RRF) beats pure semantic
and pure lexical retrieval, and all beat today's keyword search.

Configurations (same 200 held-out queries, same corpus):
  keyword_baseline   agent-style keyword search (2 most specific words, ILIKE, newest first)
  lexical_and        PostgreSQL FTS, plainto_tsquery (all terms must match)
  lexical_or         PostgreSQL FTS, OR-of-terms tsquery, ts_rank ordering
  semantic           pgvector cosine (ivfflat)
  hybrid_50_50       RRF(k=60), semantic 0.5 / lexical(or) 0.5
  hybrid_70_30       RRF, semantic weighted 0.7
  hybrid_30_70       RRF, lexical weighted 0.7
All rankers use kb_min_slots=0 so the comparison is of raw ranking quality.

Usage: python experiments/e1_retrieval.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.database import dispose_engine, get_session_factory  # noqa: E402
from app.models.embeddings import get_embedding_service  # noqa: E402
from app.services.retrieval import HybridRetriever  # noqa: E402
from evaluation.baselines import KeywordSearchBaseline  # noqa: E402
from evaluation.data import build_qrels, load_eval, load_qrels, save_result  # noqa: E402
from evaluation.metrics import bootstrap_ci, paired_bootstrap_p  # noqa: E402
from evaluation.retrieval_eval import run_queries  # noqa: E402

TOP_K = 20


HYBRID_WEIGHTS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]  # semantic weight; lexical = 1 - w


def summary_line(name, res):
    m = res["metrics"]
    return (
        f"{name:17s} nDCG@10={m['ndcg@10']:.3f} MRR={m['mrr']:.3f} P@5={m['p@5']:.3f} "
        f"Hit@5={m['hit@5']:.3f} R@10c={m['recall@10_capped']:.3f} KB-R@10={m['kb_recall@10']:.3f} "
        f"p95={res['latency_ms']['p95']}ms"
    )


async def main() -> None:
    sf = get_session_factory()
    embedder = get_embedding_service()
    eval_queries, eval_qrels = load_eval("retrieval_eval"), load_qrels()
    dev_queries = load_eval("test")
    dev_qrels = build_qrels(dev_queries)

    keyword = KeywordSearchBaseline(sf)
    await keyword.fit_idf()

    def retriever_fn(mode, lexical_mode="or", sw=0.5):
        r = HybridRetriever(
            sf, embedder, lexical_mode=lexical_mode, semantic_weight=sw, lexical_weight=round(1 - sw, 2)
        )

        async def fn(q):
            res = await r.retrieve(q, top_k=TOP_K, mode=mode, kb_min_slots=0)
            return [i.document.id for i in res.items]

        return fn

    # 1) choose the RRF weight on the DEV split (test.jsonl), never on the reported queries
    dev_results = {}
    for w in HYBRID_WEIGHTS:
        name = f"hybrid_{int(w * 100)}_{int(round(1 - w, 2) * 100)}"
        dev_results[name] = await run_queries(retriever_fn("hybrid", "or", w), dev_queries, dev_qrels)
        dev_results[name].pop("per_query")
        print("[dev]", summary_line(name, dev_results[name]))
    best_hybrid = max(dev_results, key=lambda n: dev_results[n]["metrics"]["ndcg@10"])
    best_w = HYBRID_WEIGHTS[list(dev_results).index(best_hybrid)]
    print(f"[dev] selected {best_hybrid}")

    # 2) report every configuration on the held-out retrieval_eval queries
    configs = {
        "keyword_baseline": lambda q: keyword.search(q, TOP_K),
        "lexical_and": retriever_fn("lexical", "and"),
        "lexical_or": retriever_fn("lexical", "or"),
        "semantic": retriever_fn("semantic"),
        "hybrid_50_50": retriever_fn("hybrid", "or", 0.5),
        best_hybrid: retriever_fn("hybrid", "or", best_w),
    }
    for fn in configs.values():  # warm-up
        await fn(eval_queries[0]["complaint"])

    results = {}
    for name, fn in configs.items():
        results[name] = await run_queries(fn, eval_queries, eval_qrels)
        print(summary_line(name, results[name]))

    def col(name, key):
        return [q[key] for q in results[name]["per_query"]]

    significance = {}
    for other in ["semantic", "lexical_or", "keyword_baseline"]:
        significance[f"{best_hybrid}_vs_{other}"] = {
            key: {
                "p_value_one_sided": paired_bootstrap_p(col(best_hybrid, key), col(other, key)),
                "diff": round(results[best_hybrid]["metrics"][key] - results[other]["metrics"][key], 4),
            }
            for key in ["ndcg@10", "mrr", "p@5", "hit@5", "recall@10_capped"]
        }
    cis = {n: {"ndcg@10_95ci": [round(x, 4) for x in bootstrap_ci(col(n, "ndcg@10"))]} for n in results}
    for r in results.values():
        r.pop("per_query")
    payload = {
        "n_queries": len(eval_queries),
        "n_dev_queries": len(dev_queries),
        "top_k": TOP_K,
        "weight_selection_on_dev": {"grid": HYBRID_WEIGHTS, "results": dev_results, "selected": best_hybrid},
        "results": results,
        "best_hybrid": best_hybrid,
        "significance": significance,
        "confidence_intervals": cis,
    }
    print(f"saved {save_result('e1_retrieval', payload)}")
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
