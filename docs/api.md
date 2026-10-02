# API reference

Base URL `http://localhost:8000/api/v1`. Interactive OpenAPI docs are at `/docs` and the schema at `/openapi.json`.
Every response carries `x-request-id` (pass your own to correlate logs) and `x-process-time-ms`.

## POST /tickets/resolve

Request

```json
{
  "complaint": "My broadband drops every evening around 8 and I've already restarted the router twice. I work from home and this is costing me.",
  "filters": {"product": "broadband"}
}
```

`complaint`: 10–5000 chars. `filters` (optional): restricts *ticket* candidates by `product` or `category`.

Response (abridged; field meanings below)

```json
{
  "request_id": "0b6c…",
  "understanding": {
    "intent": "connectivity_issue", "intent_confidence": 0.81,
    "intent_candidates": [["connectivity_issue", 0.81], ["speed_issue", 0.12]],
    "products": ["router", "broadband"], "severity": "high",
    "sentiment": "negative", "sentiment_score": -0.93
  },
  "resolution": {
    "steps": ["Confirm the drops only affect Wi-Fi by testing a device on an ethernet cable [1][3]", "…"],
    "summary": "…", "estimated_time": "15-20 minutes", "generator": "llm:qwen/qwen3.8-27b"
  },
  "sources": [{"source_number": 1, "type": "ticket", "id": "TKT-000123", "excerpt": "…", "relevance_score": 0.74}],
  "validation": {
    "citation_accuracy": 1.0, "citation_coverage": 1.0, "groundedness_score": 0.9,
    "claims": [{"step": 1, "claim": "…", "citations": [1, 3], "support_status": "supported",
                "signals": {"semantic": 0.82, "entailment": 0.91, "contradiction": 0.01, "lexical": 0.88}}],
    "issues": []
  },
  "confidence": {
    "heuristic_score": 0.83, "interpretation": "High - strong evidence support",
    "components": {"retrieval_quality": 0.9, "evidence_quality": 0.95, "understanding_quality": 0.81, "sufficiency_score": 1.0},
    "evidence_sufficient": true,
    "note": "Heuristic score derived from evidence signals; not a calibrated probability and not LLM self-assessment."
  },
  "decision": {"action": "RESOLVE", "reason": "High-confidence, well-grounded resolution"},
  "flags": [],
  "latency_ms": 4120,
  "stage_latency_ms": {"embed": 25, "understand_retrieve": 180, "generate": 1900, "validate": 1950, "decide": 0}
}
```

The values above illustrate the shape only; they are not measurements.

| Field | Meaning |
|---|---|
| `intent_confidence` | similarity-weighted share of the k nearest labelled tickets voting for the intent |
| `support_status` | `supported`, `weakly_supported`, `unsupported`, `contradicted`, or `uncited` |
| `relevance_score` | cosine similarity between complaint and source (absolute; used for evidence checks) |
| `generator` | `llm:<model>`, `extractive` (no LLM / fallback), or `none` (no sources) |
| `flags` | `unknown_intent`, `possible_prompt_injection`, `llm_unavailable_extractive_fallback`, `contradiction_detected`, `low_confidence` |
| `decision.action` | `RESOLVE` use the draft · `REVIEW` an agent must check it · `ESCALATE` evidence insufficient |

## POST /tickets/{request_id}/feedback

`{"rating": 1-5, "comment": "optional"}` → `{"status": "recorded"}`. 404 if the request id is unknown.
Ratings are stored next to the heuristic confidence to fit a calibration map later.

## POST /ingestion

```json
{
  "tickets": [{"ticket_id": "TKT-900001", "complaint": "…", "resolution": "…", "category": "roaming_issue",
               "product": "mobile_service", "severity": "high", "source": "crm_export", "label_source": "human_verified"}],
  "kb_articles": [{"article_id": "KB-0100", "title": "…", "content": "… (≥ 50 chars)", "category": "roaming_issue"}]
}
```

Upserts by id. Unchanged rows are skipped; changed tickets are re-embedded; changed KB articles get `version + 1`
and the previous version is archived. Response counts `*_inserted / *_updated / *_unchanged`, `embeddings_generated`,
`new_categories` (categories not previously seen), and per-row `errors` (one bad row doesn't abort the batch).

## GET / POST /taxonomy/intents

POST `{"intent_name": "roaming_issue", "description": "…", "examples": ["data roaming not working abroad"]}`
registers or updates a ticket class at runtime; the understanding service reloads its taxonomy immediately.

## GET /health

`{"status": "healthy|degraded", "database": "connected", "llm_provider": "configured (model) | not_configured (extractive fallback)", "models_loaded": true, "corpus": {"tickets": n, "kb_articles": n}, "version": "1.0.0"}`

## GET /metrics?hours=24

Request volume, decision mix, escalation rate, latency avg/p50/p95/p99, mean confidence / groundedness / citation
accuracy / coverage, LLM fallback rate, mean agent feedback. Omit `hours` for all-time.

## GET /monitoring/drift?window_hours=24

Recent window vs everything before it: unknown-intent rate, intent distribution and its total-variation distance,
mean nearest-neighbour similarity, escalation rate, plus `alerts` when thresholds (`DRIFT_*` settings) are crossed.

## Errors

All errors are JSON. Every response, including errors, carries an `x-request-id` header; quote it when reporting
a problem (it ties the response to the structured server logs).

| Status | When | Body |
|---|---|---|
| 200 + `"status": "degraded"` | `/health` when the database is unreachable or models are not loaded (the endpoint itself still answers) | health object |
| 400 | `/tickets/{request_id}/feedback` with a malformed request id (not a UUID) | `{"detail": "invalid request_id"}` |
| 404 | feedback for an unknown request id | `{"detail": "request not found"}` |
| 422 | request validation failed: complaint shorter than 10 or longer than 5,000 chars, missing fields, rating outside 1–5, invalid intent name, batch too large, invalid ticket severity | FastAPI/Pydantic format: `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` |
| 500 | unexpected server error (logged with stack trace) | `{"detail": "internal error", "request_id": "<id>"}` |
| 503 | ingestion or taxonomy endpoints called on a deployment without the ingestion service | `{"detail": "ingestion service not available"}` |

Degraded behaviour that is **not** an error: when the LLM is unreachable, rate-limited or not configured,
`/tickets/resolve` still returns 200 with an extractive draft; the response shows `resolution.generator =
"extractive"` and the flag `llm_unavailable_extractive_fallback`. Ingestion reports per-row failures in `errors`
with status 200 rather than failing the whole batch.

**Authentication:** none in this build (internal tool behind the agent desktop). For production, put the service
behind an API gateway with OAuth2/JWT and per-client rate limits (see ARCHITECTURE.md §6, "Security and privacy").
