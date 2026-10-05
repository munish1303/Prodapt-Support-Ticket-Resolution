"""Experiment 9 - Per-request wrong-scenario detection ("grounded but wrong", EVALUATION.md §4.1, §6.3, §6.4).

Question: can runtime signals tell when the retrieved context is about a *different* problem than the complaint
(a confusable scenario or an issue type the corpus has never seen), so that such drafts go to REVIEW instead of
being auto-resolved?

Two phases:
  collect  (needs the database and models; run in the API container) - for every complaint in the tuning and test
           sets, run understanding + hybrid retrieval exactly as production does, and record per-source signals:
           semantic / lexical ranks, cosine relevance, category, product, and a cross-encoder score
           (cross-encoder/ms-marco-MiniLM-L-6-v2) between the complaint and each source. Ground-truth scenario ids
           are stored for *labelling only*; no signal uses them.
  analyze  (no database) - derive candidate signals, choose a detector on the tuning set only, then measure it on
           the untouched draft-level test sets (generation cases with LLM and extractive drafts, the 60 unseen-issue
           complaints, and the 50 AI-rated drafts).

Data:
  tuning   data/evaluation/understanding_eval.jsonl (500 held-out complaints, 40 scenarios)
  check    data/evaluation/retrieval_eval.jsonl (200, retrieval-level held-out check)
  test     data/evaluation/generation_eval.jsonl (100) and novel_intent_eval.jsonl (60), joined with the stored
           runs in experiments/results/generation_llm.json / generation_extractive.json / ai_rater_eval.json
All four sets are disjoint (checked when written).

Usage:
  python experiments/e9_wrong_scenario.py collect   # -> experiments/results/e9_features.jsonl
  python experiments/e9_wrong_scenario.py analyze   # -> experiments/results/e9_wrong_scenario.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.data import RESULTS_DIR, load_eval  # noqa: E402

FEATURES = RESULTS_DIR / "e9_features.jsonl"
SETS = {
    "tune": "understanding_eval",
    "check": "retrieval_eval",
    "gen": "generation_eval",
    "novel": "novel_intent_eval",
}
# The problem statement's own example (EVALUATION.md §6.3); its true scenario is evening Wi-Fi drops / congestion.
EXAMPLE = {
    "id": "EX-6.3",
    "complaint": "My broadband drops every evening around 8 and I've already restarted the router twice, "
    "I work from home and this is costing me",
    "scenario_id": "CONN_WIFI_EVENING_INTERFERENCE",
}
CANDIDATES = 70  # all hybrid candidates (50 tickets + 20 KB articles), so full semantic / lexical ranks are known


async def collect() -> None:
    from app.api.v1.dependencies import build_container
    from app.config import settings
    from app.core.database import dispose_engine
    from app.services.retrieval import HybridRetriever, Reranker

    container = await build_container()
    retriever = HybridRetriever(container.session_factory, container.embedder)
    cross = Reranker()  # only its .model (CrossEncoder) is used: scores, not reordering

    done: set[str] = set()
    if FEATURES.exists():
        done = {json.loads(line)["id"] for line in FEATURES.open(encoding="utf-8")}
    cases = [(name, c) for name, f in SETS.items() for c in load_eval(f)] + [("example", EXAMPLE)]
    todo = [(n, c) for n, c in cases if c["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)

    for k, (set_name, case) in enumerate(todo, 1):
        complaint = case["complaint"]
        emb = await asyncio.to_thread(container.embedder.encode, complaint)
        meta = await container.understanding.analyze(complaint, emb)
        prod = await retriever.retrieve(
            complaint, settings.RETRIEVAL_TOP_K, "hybrid", emb, None, settings.RETRIEVAL_KB_MIN_SLOTS
        )
        full = await retriever.retrieve(complaint, CANDIDATES, "hybrid", emb, None, 0)
        kb_cands = [i for i in full.items if i.document.doc_type == "kb_article"]
        pairs = [[complaint, i.document.text[:2000]] for i in prod.items + kb_cands]
        scores = cross.model.predict(pairs, show_progress_bar=False).tolist() if pairs else []
        ce_prod, ce_kb = scores[: len(prod.items)], scores[len(prod.items) :]

        def src(item, ce):
            d = item.document
            return {
                "id": d.id,
                "type": d.doc_type,
                "category": d.metadata.get("category"),
                "product": d.metadata.get("product"),
                "scenario_id": d.metadata.get("scenario_id"),  # label only
                "relevance": round(item.relevance, 4),
                "sem_rank": item.semantic_rank,
                "lex_rank": item.lexical_rank,
                "ce": round(float(ce), 4),
            }

        row = {
            "id": case["id"],
            "set": set_name,
            "scenario_id": case.get("scenario_id"),  # label only
            "category": case.get("category"),
            "intent": meta.intent,
            "intent_confidence": round(meta.intent_confidence, 4),
            "intent_unknown": meta.is_unknown_intent,
            "nearest_similarity": meta.nearest_similarity,
            "products": meta.products,
            "top": [src(i, s) for i, s in zip(prod.items, ce_prod)],
            "kb_candidates": [src(i, s) for i, s in zip(kb_cands, ce_kb)],
            "sem_top10": [i.document.id for i in sorted(full.items, key=lambda i: i.semantic_rank or 10**6)[:10]],
            "lex_top10": [i.document.id for i in sorted(full.items, key=lambda i: i.lexical_rank or 10**6)[:10]],
        }
        with FEATURES.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if k % 50 == 0:
            print(f"{k}/{len(todo)}", flush=True)
    await dispose_engine()
    print("collected", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("phase", choices=["collect", "analyze"])
    args = parser.parse_args()
    if args.phase == "collect":
        asyncio.run(collect())
    else:
        from experiments.e9_analysis import analyze

        analyze()


if __name__ == "__main__":
    main()
