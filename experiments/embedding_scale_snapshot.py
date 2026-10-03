"""Record the similarity scale the live system produces, so scale-dependent thresholds can be carried over when the
embedding model changes (see experiments/recalibrate_thresholds.py).

For every complaint in the dev split (data/evaluation/test.jsonl, never used for reporting) and the novel-class set
(roaming complaints absent from the corpus, which cover the low-similarity tail), it records what the production
code computes:
  top1_relevance      cosine of the best retrieved source       (EVIDENCE_MIN_TOP_RELEVANCE)
  top5_mean_relevance mean cosine of the top-5 sources          (EVIDENCE_MIN_AVG_RELEVANCE, CONF_RETRIEVAL_NORMALISER)
  nearest_similarity  cosine of the nearest labelled ticket      (DRIFT_LOW_SIMILARITY_ALERT)

Usage: python experiments/embedding_scale_snapshot.py --tag minilm     (run against the database as currently embedded)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.core.database import dispose_engine, get_session_factory  # noqa: E402
from app.models.embeddings import get_embedding_service  # noqa: E402
from app.services.retrieval import HybridRetriever, RetrievalService  # noqa: E402
from app.services.understanding import PgNeighbourIndex  # noqa: E402
from evaluation.data import load_eval, save_result  # noqa: E402

STATS = ["top1_relevance", "top5_mean_relevance", "nearest_similarity"]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    embedder = get_embedding_service()
    sf = get_session_factory()
    retrieval = RetrievalService(HybridRetriever(sf, embedder), None, use_reranking=False)
    index = PgNeighbourIndex(sf)
    out: dict[str, dict[str, list[float]]] = {}
    for name, items in (("dev", load_eval("test")), ("novel", load_eval("novel_intent_eval"))):
        rows: dict[str, list[float]] = {k: [] for k in STATS}
        for it in items:
            emb = embedder.encode(it["complaint"])
            res = await retrieval.retrieve(it["complaint"], query_embedding=emb)
            rel = sorted(res.relevance_scores, reverse=True)
            nn = await index.nearest(emb, settings.INTENT_KNN_K)
            rows["top1_relevance"].append(round(float(rel[0]), 4) if rel else 0.0)
            rows["top5_mean_relevance"].append(round(float(np.mean(rel[:5])), 4) if rel else 0.0)
            rows["nearest_similarity"].append(round(max(n.similarity for n in nn), 4) if nn else 0.0)
        out[name] = rows
        print(name, {k: round(float(np.median(v)), 3) for k, v in rows.items()})
    await dispose_engine()
    payload = {
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dim": settings.EMBEDDING_DIM,
        "n": {k: len(v["top1_relevance"]) for k, v in out.items()},
        "values": out,
    }
    print(f"saved {save_result(f'embedding_scale_{args.tag}', payload)}")


if __name__ == "__main__":
    asyncio.run(main())
