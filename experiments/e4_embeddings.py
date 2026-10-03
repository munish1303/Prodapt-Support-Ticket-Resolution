"""Experiment 4 - Embedding model comparison (plan §35.1, SHOULD RUN).

Question: would a larger or newer embedding model retrieve better than all-MiniLM-L6-v2, and at what cost?

Method: same corpus as production (data/processed: 2,043 tickets embedded as "complaint resolution", 40 KB articles
as "title\\ncontent"), same 200 held-out queries and graded judgments as E1 (data/evaluation/retrieval_qrels.tsv).
Each model embeds corpus and queries (L2-normalised); ranking is exact cosine top-20 in memory, so the comparison
isolates the embedding model (no ivfflat approximation, no lexical fusion). Metrics are E1's (evaluation/
retrieval_eval.py); significance is a paired bootstrap against the baseline. Cost: dimensions, one-query encode
latency on CPU, corpus encode time.

Models: all-MiniLM-L6-v2 (384-d, deployed), all-mpnet-base-v2 (768-d, plan), BAAI/bge-small-en-v1.5 (384-d, a newer
small model that would be a drop-in replacement with no schema change). OpenAI text-embedding-3-small (in the plan)
is not run: it needs a paid API key.

Decision rule (plan): keep all-MiniLM-L6-v2 unless another model is significantly better.

Usage: python experiments/e4_embeddings.py [--models m1,m2,...]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentence_transformers import SentenceTransformer  # noqa: E402

from evaluation.data import PROCESSED_DIR, load_eval, load_qrels, read_jsonl, save_result  # noqa: E402
from evaluation.metrics import paired_bootstrap_p, percentile  # noqa: E402
from evaluation.retrieval_eval import aggregate, per_query_metrics  # noqa: E402

BASELINE = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODELS = [BASELINE, "sentence-transformers/all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"]
TOP_K = 20
COMPARE = ["ndcg@10", "mrr", "p@5", "hit@5", "recall@10_capped", "kb_recall@10"]


def corpus() -> tuple[list[str], list[str]]:
    ids, texts = [], []
    for t in read_jsonl(PROCESSED_DIR / "tickets.jsonl"):
        ids.append(t["ticket_id"])
        texts.append(f"{t['complaint']} {t.get('resolution') or ''}".strip())
    for a in read_jsonl(PROCESSED_DIR / "kb_articles.jsonl"):
        ids.append(a["article_id"])
        texts.append(f"{a['title']}\n{a['content']}")
    return ids, texts


def run_model(name: str, doc_ids: list[str], doc_texts: list[str], queries: list[dict], qrels: dict) -> dict:
    model = SentenceTransformer(name, device="cpu")
    t0 = time.perf_counter()
    docs = model.encode(doc_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    corpus_s = time.perf_counter() - t0

    model.encode(["warm-up"], normalize_embeddings=True)
    per_query, lat = [], []
    for q in queries:
        t1 = time.perf_counter()
        qv = model.encode([q["complaint"]], normalize_embeddings=True, show_progress_bar=False)[0]
        lat.append((time.perf_counter() - t1) * 1000)
        top = np.argsort(-(docs @ qv))[:TOP_K]
        per_query.append(per_query_metrics([doc_ids[i] for i in top], qrels.get(q["id"], {})))
    return {
        "dims": int(docs.shape[1]),
        "metrics": aggregate(per_query),
        "query_encode_ms": {"p50": round(percentile(lat, 50), 1), "p95": round(percentile(lat, 95), 1)},
        "corpus_encode_s": round(corpus_s, 1),
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if BASELINE not in models:
        models.insert(0, BASELINE)

    doc_ids, doc_texts = corpus()
    queries, qrels = load_eval("retrieval_eval"), load_qrels()
    results: dict[str, dict] = {}
    for name in models:
        print(f"embedding with {name} ...", flush=True)
        results[name] = run_model(name, doc_ids, doc_texts, queries, qrels)
        m = results[name]["metrics"]
        print(
            f"  {results[name]['dims']}-d  nDCG@10={m['ndcg@10']:.3f} MRR={m['mrr']:.3f} P@5={m['p@5']:.3f} "
            f"Hit@5={m['hit@5']:.3f} R@10c={m['recall@10_capped']:.3f}  "
            f"query p50={results[name]['query_encode_ms']['p50']} ms  corpus={results[name]['corpus_encode_s']} s",
            flush=True,
        )

    base = results[BASELINE]["per_query"]
    significance = {}
    for name in models:
        if name == BASELINE:
            continue
        other = results[name]["per_query"]
        significance[name] = {
            k: {
                "diff": round(float(np.mean([o[k] for o in other]) - np.mean([b[k] for b in base])), 4),
                # one-sided paired bootstrap: small p_better = significantly better than the baseline,
                # small p_worse = significantly worse
                "p_better": round(paired_bootstrap_p([o[k] for o in other], [b[k] for b in base]), 4),
                "p_worse": round(paired_bootstrap_p([b[k] for b in base], [o[k] for o in other]), 4),
            }
            for k in COMPARE
        }
    payload = {
        "experiment": "E4 embedding model comparison",
        "corpus": {
            "tickets": sum(1 for d in doc_ids if d.startswith("TKT-")),
            "kb_articles": sum(1 for d in doc_ids if d.startswith("KB-")),
        },
        "queries": len(queries),
        "ranking": "exact cosine top-20 in memory (isolates the embedding model)",
        "results": {n: {k: v for k, v in r.items() if k != "per_query"} for n, r in results.items()},
        "significance_vs_baseline": significance,
    }
    print(f"saved {save_result('e4_embeddings', payload)}")


if __name__ == "__main__":
    main()
