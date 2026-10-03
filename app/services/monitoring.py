"""Operational metrics and drift signals computed from the resolution_requests log."""

from __future__ import annotations

from sqlalchemy import text

from app.config import settings

# Embedding model of request-log rows written before the model name was logged (the original default).
LEGACY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


async def get_metrics(session, hours: int | None = None) -> dict:
    window = "" if hours is None else "WHERE created_at >= NOW() - make_interval(hours => :h)"
    params = {} if hours is None else {"h": hours}
    row = (
        await session.execute(
            text(f"""
        SELECT count(*),
               avg(latency_ms),
               percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms),
               percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms),
               percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms),
               avg(heuristic_confidence), avg(groundedness_score), avg(citation_accuracy), avg(citation_coverage),
               avg(CASE WHEN generator = 'extractive' AND flags ? 'llm_unavailable_extractive_fallback' THEN 1.0 ELSE 0.0 END),
               avg(feedback_rating)
        FROM resolution_requests {window}
    """),
            params,
        )
    ).one()
    decisions = (
        await session.execute(
            text(f"SELECT decision, count(*) FROM resolution_requests {window} GROUP BY decision"), params
        )
    ).all()
    total = int(row[0] or 0)

    def f(x):
        return None if x is None else round(float(x), 4)

    return {
        "window_hours": hours,
        "requests_total": total,
        "requests_by_decision": {d: int(c) for d, c in decisions},
        "escalation_rate": f(sum(c for d, c in decisions if d == "ESCALATE") / total) if total else None,
        "latency_ms": {"avg": f(row[1]), "p50": f(row[2]), "p95": f(row[3]), "p99": f(row[4])},
        "avg_confidence": f(row[5]),
        "avg_groundedness": f(row[6]),
        "avg_citation_accuracy": f(row[7]),
        "avg_citation_coverage": f(row[8]),
        "llm_fallback_rate": f(row[9]),
        "avg_feedback_rating": f(row[10]),
    }


async def get_drift_report(session, window_hours: int | None = None) -> dict:
    """Compare the recent window with everything before it.

    Signals: unknown-intent rate, intent mix shift (total variation distance),
    mean nearest-neighbour similarity (a falling value means incoming complaints
    look less like anything in history), and escalation rate.
    """
    h = window_hours or settings.DRIFT_WINDOW_HOURS
    rows = (
        await session.execute(
            text("""
        SELECT (created_at >= NOW() - make_interval(hours => :h)) AS recent,
               extracted_metadata->>'intent' AS intent,
               (extracted_metadata->>'nearest_similarity')::float AS nn_sim,
               decision
        FROM resolution_requests
        -- Similarities are only comparable within one embedding model, so the baseline restarts when the model
        -- changes. Rows logged before the model was recorded all came from the legacy default.
        WHERE coalesce(extracted_metadata->>'embedding_model', :legacy) = :model
    """),
            {"h": h, "legacy": LEGACY_EMBEDDING_MODEL, "model": settings.EMBEDDING_MODEL},
        )
    ).all()
    windows = {"recent": [r for r in rows if r[0]], "baseline": [r for r in rows if not r[0]]}

    def summarise(rs):
        n = len(rs)
        if n == 0:
            return {"n": 0}
        intents: dict[str, int] = {}
        for r in rs:
            intents[r[1]] = intents.get(r[1], 0) + 1
        sims = [r[2] for r in rs if r[2] is not None]
        return {
            "n": n,
            "unknown_intent_rate": round(intents.get("unknown_intent", 0) / n, 4),
            "intent_distribution": {k: round(v / n, 4) for k, v in sorted(intents.items())},
            "mean_nearest_similarity": round(sum(sims) / len(sims), 4) if sims else None,
            "escalation_rate": round(sum(1 for r in rs if r[3] == "ESCALATE") / n, 4),
        }

    recent, baseline = summarise(windows["recent"]), summarise(windows["baseline"])
    tvd = intent_tvd(recent, baseline)
    return {
        "window_hours": h,
        "recent": recent,
        "baseline": baseline,
        "intent_tvd": tvd,
        "intent_tvd_alert_threshold": tvd_threshold(recent, baseline),
        "alerts": drift_alerts(recent, baseline, tvd),
    }


def intent_tvd(recent: dict, baseline: dict) -> float | None:
    """Total variation distance between the recent and baseline intent distributions."""
    if not (recent.get("n") and baseline.get("n")):
        return None
    r, b = recent["intent_distribution"], baseline["intent_distribution"]
    return round(0.5 * sum(abs(r.get(k, 0) - b.get(k, 0)) for k in set(r) | set(b)), 4)


def tvd_threshold(recent: dict, baseline: dict) -> float | None:
    """Sampling noise in TVD shrinks like 1/sqrt(n); coef/sqrt(min window) approximates its 99th percentile
    (coefficient measured on held-out traffic, see EVALUATION.md section 6). None if a window is too small."""
    n = min(recent.get("n", 0), baseline.get("n", 0))
    if n < settings.DRIFT_MIN_WINDOW:
        return None
    return round(settings.DRIFT_TVD_ALERT_COEF / n**0.5, 4)


def drift_alerts(recent: dict, baseline: dict, tvd: float | None) -> list[str]:
    alerts = []
    if recent.get("n"):
        if recent["unknown_intent_rate"] > settings.DRIFT_UNKNOWN_RATE_ALERT:
            alerts.append(
                f"unknown_intent_rate {recent['unknown_intent_rate']} > {settings.DRIFT_UNKNOWN_RATE_ALERT}: "
                "possible new ticket class; review unknown-intent requests and extend the taxonomy"
            )
        sim = recent.get("mean_nearest_similarity")
        if sim is not None and sim < settings.DRIFT_LOW_SIMILARITY_ALERT:
            alerts.append(
                f"mean nearest-neighbour similarity {sim} < {settings.DRIFT_LOW_SIMILARITY_ALERT}: "
                "incoming complaints are unlike historical tickets"
            )
    threshold = tvd_threshold(recent, baseline)
    if tvd is not None and threshold is not None and tvd > threshold:
        alerts.append(
            f"intent mix shifted: TVD {tvd} > {threshold} (noise threshold for these window sizes); "
            "a new issue type may be landing in existing classes - review recent requests"
        )
    return alerts


async def record_feedback(session, request_id: str, rating: int, comment: str | None) -> bool:
    result = await session.execute(
        text(
            "UPDATE resolution_requests SET feedback_rating = :r, feedback_text = :t WHERE request_id = CAST(:id AS uuid)"
        ),
        {"r": rating, "t": comment, "id": request_id},
    )
    await session.commit()
    return result.rowcount > 0
