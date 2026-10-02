"""Small helpers shared by experiment scripts."""

from __future__ import annotations

from collections import Counter

from evaluation.metrics import mean


def phase_summary(rows: list[dict], target_intent: str) -> dict:
    n = len(rows) or 1
    return {
        "intent_distribution": dict(Counter(r["intent"] for r in rows)),
        "target_intent_accuracy": round(sum(r["intent"] == target_intent for r in rows) / n, 4),
        "unknown_intent_rate": round(sum(r["intent"] == "unknown_intent" for r in rows) / n, 4),
        "decisions": dict(Counter(r["decision"] for r in rows)),
        "resolve_rate": round(sum(r["decision"] == "RESOLVE" for r in rows) / n, 4),
        "mean_confidence": round(mean([r["confidence"] for r in rows]), 4),
        "mean_nearest_similarity": round(mean([r["nearest_similarity"] or 0 for r in rows]), 4),
        "mean_max_relevance": round(mean([r["max_relevance"] for r in rows]), 4),
        "target_kb_retrieved_rate": round(sum(r["roaming_kb_retrieved"] for r in rows) / n, 4),
        "mean_groundedness": round(mean([r["groundedness"] for r in rows]), 4),
    }
