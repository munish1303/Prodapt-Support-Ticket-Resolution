"""Understanding evaluation: intent, products, severity, sentiment (+ unknown-intent / evolving classes).

Sets
  data/evaluation/understanding_eval.jsonl  500 held-out complaints   -> reported metrics
  data/evaluation/test.jsonl                199 held-out complaints   -> dev set for the unknown-intent threshold sweep
  data/evaluation/novel_intent_eval.jsonl   60 wave-2 'roaming_issue' complaints (class absent at launch)

Index
  --index memory : exact k-NN over data/processed tickets (no DB needed; same embeddings as the DB)
  --index pg     : the production pgvector index (DB must be ingested)

Usage: python evaluation/understanding_eval.py [--index memory|pg] [--classifier knn|llm]
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.llm import build_llm_provider  # noqa: E402
from app.models.embeddings import get_embedding_service  # noqa: E402
from app.services.understanding import (  # noqa: E402
    SEVERITY_LEVELS,
    UNKNOWN_INTENT,
    InMemoryNeighbourIndex,
    IntentDef,
    NeighbourIndex,
    SentimentAnalyzer,
    UnderstandingService,
    knn_vote,
    load_taxonomy_files,
)
from evaluation.data import PROCESSED_DIR, load_eval, read_jsonl, save_result  # noqa: E402

ROAMING = IntentDef(
    "roaming_issue",
    "Problems using mobile service while abroad: roaming data, calls or charges",
    ["data roaming not working abroad", "roaming charges on my bill", "can't call while travelling overseas"],
)


def build_memory_index(embedder, files: list[str]) -> InMemoryNeighbourIndex:
    rows = [r for f in files for r in read_jsonl(PROCESSED_DIR / f)]
    vecs = embedder.encode([r["complaint"] for r in rows], batch_size=64)
    return InMemoryNeighbourIndex(
        vecs,
        [r["category"] for r in rows],
        [r["severity"] for r in rows],
        [r["metadata"].get("products", [r["product"]]) for r in rows],
    )


def product_prf(true_sets, pred_sets) -> dict:
    tp = sum(len(t & p) for t, p in zip(true_sets, pred_sets))
    fp = sum(len(p - t) for t, p in zip(true_sets, pred_sets))
    fn = sum(len(t - p) for t, p in zip(true_sets, pred_sets))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "micro_precision": round(prec, 4),
        "micro_recall": round(rec, 4),
        "micro_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "exact_match": round(sum(t == p for t, p in zip(true_sets, pred_sets)) / len(true_sets), 4),
        "primary_product_recall": None,
    }


async def run_service(service: UnderstandingService, items: list[dict]) -> tuple[list, list[float]]:
    preds, lat = [], []
    for it in items:
        t0 = time.perf_counter()
        preds.append(await service.analyze(it["complaint"]))
        lat.append((time.perf_counter() - t0) * 1000)
    return preds, lat


def score(items, preds) -> dict:
    y_int = [it["category"] for it in items]
    p_int = [p.intent for p in preds]
    labels = sorted(set(y_int))
    report = classification_report(y_int, p_int, labels=labels, output_dict=True, zero_division=0)
    known = [(t, p) for t, p in zip(y_int, p_int) if p != UNKNOWN_INTENT]
    sev_true = [it["severity"] for it in items]
    sev_pred = [p.severity for p in preds]
    sent_true = [it["sentiment"] for it in items]
    sent_pred = [p.sentiment for p in preds]
    prod = product_prf([set(it["products"]) for it in items], [set(p.products) for p in preds])
    prod["primary_product_recall"] = round(
        sum(it["product"] in p.products for it, p in zip(items, preds)) / len(items), 4
    )
    return {
        "n": len(items),
        "intent": {
            "accuracy": round(sum(t == p for t, p in zip(y_int, p_int)) / len(items), 4),
            "macro_f1": round(f1_score(y_int, p_int, labels=labels, average="macro", zero_division=0), 4),
            "false_unknown_rate": round(p_int.count(UNKNOWN_INTENT) / len(items), 4),
            "accuracy_when_not_unknown": round(sum(t == p for t, p in known) / len(known), 4) if known else None,
            "per_class_f1": {lbl: round(report[lbl]["f1-score"], 4) for lbl in labels},
        },
        "products": prod,
        "severity": {
            "accuracy": round(sum(t == p for t, p in zip(sev_true, sev_pred)) / len(items), 4),
            "within_one_level": round(
                sum(abs(SEVERITY_LEVELS.index(t) - SEVERITY_LEVELS.index(p)) <= 1 for t, p in zip(sev_true, sev_pred))
                / len(items),
                4,
            ),
            "macro_f1": round(
                f1_score(sev_true, sev_pred, labels=SEVERITY_LEVELS, average="macro", zero_division=0), 4
            ),
            "confusion_matrix": {
                "labels": SEVERITY_LEVELS,
                "matrix": confusion_matrix(sev_true, sev_pred, labels=SEVERITY_LEVELS).tolist(),
            },
        },
        "sentiment": {
            "accuracy": round(sum(t == p for t, p in zip(sent_true, sent_pred)) / len(items), 4),
            "macro_f1": round(
                f1_score(
                    sent_true, sent_pred, labels=["negative", "neutral", "positive"], average="macro", zero_division=0
                ),
                4,
            ),
            "confusion_matrix": {
                "labels": ["negative", "neutral", "positive"],
                "matrix": confusion_matrix(sent_true, sent_pred, labels=["negative", "neutral", "positive"]).tolist(),
            },
            "pred_distribution": dict(Counter(sent_pred)),
        },
    }


async def unknown_threshold_sweep(service, index, dev_items, novel_items) -> list[dict]:
    """Trade-off between flagging in-distribution complaints as unknown (bad) and catching novel ones (good)."""
    embedder = service.embedder

    async def neighbours(items):
        out = []
        for it in items:
            emb = embedder.encode(it["complaint"])
            nn = [(n.category, n.similarity) for n in await index.nearest(emb, settings.INTENT_KNN_K)]
            nn += list(zip(service._proto_labels, (service._proto_emb @ emb).tolist()))
            out.append(nn)
        return out

    dev_nn, novel_nn = await neighbours(dev_items), await neighbours(novel_items)
    rows = []
    for min_conf in [0.0, 0.35, 0.45, 0.55, 0.65, 0.75]:
        for min_sim in [0.0, 0.4, 0.5, 0.55, 0.6, 0.65]:
            dev_unknown = np.mean([knn_vote(n, min_conf, min_sim)["is_unknown"] for n in dev_nn])
            dev_correct = np.mean(
                [knn_vote(n, min_conf, min_sim)["intent"] == it["category"] for n, it in zip(dev_nn, dev_items)]
            )
            novel_flagged = np.mean([knn_vote(n, min_conf, min_sim)["is_unknown"] for n in novel_nn])
            rows.append(
                {
                    "min_confidence": min_conf,
                    "min_similarity": min_sim,
                    "dev_false_unknown_rate": round(float(dev_unknown), 4),
                    "dev_accuracy": round(float(dev_correct), 4),
                    "novel_detection_rate": round(float(novel_flagged), 4),
                }
            )
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=["memory", "pg"], default="memory")
    parser.add_argument("--classifier", choices=["knn", "llm"], default="knn")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    embedder = get_embedding_service()
    intents, products = load_taxonomy_files()
    sentiment = SentimentAnalyzer()
    llm = build_llm_provider() if args.classifier == "llm" else None

    if args.index == "pg":
        from app.core.database import dispose_engine, get_session_factory
        from app.services.understanding import PgNeighbourIndex

        index: NeighbourIndex = PgNeighbourIndex(get_session_factory())
    else:
        index = build_memory_index(embedder, ["tickets.jsonl"])

    service = UnderstandingService(embedder, index, intents, products, sentiment, llm, args.classifier)
    items = load_eval("understanding_eval")[: args.limit]
    novel = load_eval("novel_intent_eval")
    dev = load_eval("test")

    preds, lat = await run_service(service, items)
    result: dict[str, Any] = {
        "config": {
            "index": args.index,
            "classifier": args.classifier,
            "knn_k": settings.INTENT_KNN_K,
            "min_confidence": settings.INTENT_MIN_CONFIDENCE,
            "min_similarity": settings.INTENT_MIN_SIMILARITY,
            "severity_sentiment_bump": settings.SEVERITY_NEGATIVE_SENTIMENT_BUMP,
        },
        "understanding": score(items, preds),
        "latency_ms": {"mean": round(float(np.mean(lat)), 1), "p95": round(float(np.percentile(lat, 95)), 1)},
    }

    # Novel class before it exists in the corpus/taxonomy: should be flagged unknown.
    novel_preds, _ = await run_service(service, novel)
    before = Counter(p.intent for p in novel_preds)
    result["novel_intent_before_ingestion"] = {
        "n": len(novel),
        "flagged_unknown_rate": round(before.get(UNKNOWN_INTENT, 0) / len(novel), 4),
        "predicted_distribution": dict(before),
    }

    if args.index == "memory" and args.classifier == "knn":
        result["unknown_threshold_sweep_on_dev"] = await unknown_threshold_sweep(service, index, dev, novel)
        # Evolving classes: register the new intent + ingest wave-2 tickets, re-evaluate.
        evolved = build_memory_index(embedder, ["tickets.jsonl", "tickets_wave2.jsonl"])
        service2 = UnderstandingService(embedder, evolved, intents + [ROAMING], products, sentiment, None, "knn")
        after_preds, _ = await run_service(service2, novel)
        after = Counter(p.intent for p in after_preds)
        regress_preds, _ = await run_service(service2, items)
        result["novel_intent_after_ingestion"] = {
            "n": len(novel),
            "accuracy_roaming_issue": round(after.get("roaming_issue", 0) / len(novel), 4),
            "predicted_distribution": dict(after),
            "in_distribution_intent_accuracy_after": score(items, regress_preds)["intent"]["accuracy"],
        }

    tag = f"understanding_{args.classifier}_{args.index}"
    path = save_result(tag, result)
    u = result["understanding"]
    print(
        f"intent acc={u['intent']['accuracy']} macroF1={u['intent']['macro_f1']} false_unknown={u['intent']['false_unknown_rate']}"
    )
    print(f"products microF1={u['products']['micro_f1']} exact={u['products']['exact_match']}")
    print(f"severity acc={u['severity']['accuracy']} within1={u['severity']['within_one_level']}")
    print(f"sentiment acc={u['sentiment']['accuracy']} macroF1={u['sentiment']['macro_f1']}")
    print("novel before:", result["novel_intent_before_ingestion"])
    if "novel_intent_after_ingestion" in result:
        print("novel after:", result["novel_intent_after_ingestion"])
    print(f"saved {path}")
    if args.index == "pg":
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
