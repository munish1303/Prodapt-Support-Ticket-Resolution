"""Operational metrics and drift signals computed from the resolution_requests log."""

from __future__ import annotations

from sqlalchemy import text

from app.config import settings


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
    """),
            {"h": h},
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
    alerts = []
    tvd = None
    if recent.get("n") and baseline.get("n"):
        keys = set(recent["intent_distribution"]) | set(baseline["intent_distribution"])
        tvd = round(
            0.5
            * sum(
                abs(recent["intent_distribution"].get(k, 0) - baseline["intent_distribution"].get(k, 0)) for k in keys
            ),
            4,
        )
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
    return {"window_hours": h, "recent": recent, "baseline": baseline, "intent_tvd": tvd, "alerts": alerts}


async def record_feedback(session, request_id: str, rating: int, comment: str | None) -> bool:
    result = await session.execute(
        text(
            "UPDATE resolution_requests SET feedback_rating = :r, feedback_text = :t WHERE request_id = CAST(:id AS uuid)"
        ),
        {"r": rating, "t": comment, "id": request_id},
    )
    await session.commit()
    return result.rowcount > 0
