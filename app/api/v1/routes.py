"""API v1 routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.dependencies import Container, get_container
from app.api.v1.models import (
    ClaimValidationResponse,
    ConfidenceResponse,
    DecisionResponse,
    FeedbackRequest,
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    IntentCreateRequest,
    ResolutionResponse,
    SourceResponse,
    TicketResolutionRequest,
    TicketResolutionResponse,
    UnderstandingResponse,
    ValidationResponse,
)
from app.config import settings
from app.core.database import get_db
from app.services import monitoring
from app.services.decision import interpret_confidence
from app.services.ingestion import IngestionReport, IngestionService
from app.services.pipeline import PipelineResult
from app.services.understanding import IntentDef

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def _require_ingestion(container: Container) -> IngestionService:
    if container.ingestion is None:
        raise HTTPException(status_code=503, detail="ingestion service not available")
    return container.ingestion


def _excerpt(text_: str, n: int = 280) -> str:
    text_ = " ".join(text_.split())
    return text_ if len(text_) <= n else text_[: n - 1] + "…"


def to_response(r: PipelineResult) -> TicketResolutionResponse:
    return TicketResolutionResponse(
        request_id=r.request_id,
        understanding=UnderstandingResponse(
            intent=r.metadata.intent,
            intent_confidence=r.metadata.intent_confidence,
            intent_candidates=r.metadata.intent_candidates,
            products=r.metadata.products,
            severity=r.metadata.severity,
            sentiment=r.metadata.sentiment,
            sentiment_score=r.metadata.sentiment_score,
        ),
        resolution=ResolutionResponse(
            steps=r.generation.resolution_steps,
            summary=r.generation.summary,
            estimated_time=r.generation.estimated_time,
            generator=r.generation.generator,
        ),
        sources=[
            SourceResponse(
                source_number=i,
                type=item.document.doc_type,
                id=item.document.id,
                title=item.document.metadata.get("title"),
                excerpt=_excerpt(item.document.text),
                relevance_score=round(item.relevance, 4),
                semantic_rank=item.semantic_rank,
                lexical_rank=item.lexical_rank,
            )
            for i, item in enumerate(r.retrieval.items, start=1)
        ],
        validation=ValidationResponse(
            citation_accuracy=r.validation.citation_accuracy,
            citation_coverage=r.validation.citation_coverage,
            groundedness_score=r.validation.groundedness_score,
            claims=[
                ClaimValidationResponse(
                    step=n, claim=c.claim, citations=c.citations, support_status=c.support_status, signals=c.methods
                )
                for n, c in enumerate(r.validation.claims, start=1)
            ],
            issues=r.validation.issues,
        ),
        confidence=ConfidenceResponse(
            heuristic_score=r.decision.heuristic_confidence,
            interpretation=interpret_confidence(r.decision.heuristic_confidence),
            components=r.decision.confidence_components,
            evidence_sufficient=r.decision.evidence_sufficient,
        ),
        decision=DecisionResponse(action=r.decision.decision, reason=r.decision.decision_reason),
        flags=r.flags,
        latency_ms=r.latency_ms,
        stage_latency_ms=r.stage_ms,
    )


@router.post("/tickets/resolve", response_model=TicketResolutionResponse, tags=["resolution"])
async def resolve_ticket(req: TicketResolutionRequest, container: Container = Depends(get_container)):
    """Paste a raw customer complaint; get structured understanding, a cited resolution draft,
    validation results, an evidence-derived confidence score and a RESOLVE/REVIEW/ESCALATE decision."""
    result = await container.pipeline.process(req.complaint, metadata_filters=req.filters)
    return to_response(result)


@router.post("/tickets/{request_id}/feedback", tags=["resolution"])
async def submit_feedback(request_id: str, req: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    """Agent feedback on a draft; collected to calibrate the heuristic confidence later."""
    try:
        found = await monitoring.record_feedback(db, request_id, req.rating, req.comment)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid request_id") from exc
    if not found:
        raise HTTPException(status_code=404, detail="request not found")
    return {"status": "recorded"}


@router.post("/ingestion", response_model=IngestionResponse, tags=["ingestion"])
async def ingest(req: IngestionRequest, container: Container = Depends(get_container)):
    """Incrementally add or update resolved tickets and KB articles (embeddings generated on the fly)."""
    ingestion = _require_ingestion(container)
    report = IngestionReport()
    await ingestion.ingest_tickets(req.tickets, report)
    await ingestion.ingest_kb_articles(req.kb_articles, report)
    return IngestionResponse(**report.__dict__)


@router.post("/taxonomy/intents", tags=["ingestion"])
async def add_intent(req: IntentCreateRequest, container: Container = Depends(get_container)):
    """Register a new (or update an existing) ticket class at runtime."""
    await _require_ingestion(container).upsert_intents([IntentDef(req.intent_name, req.description, req.examples)])
    await container.refresh_taxonomy()
    return {"status": "ok", "intents": [i.intent_name for i in container.understanding.intents]}


@router.get("/taxonomy/intents", tags=["ingestion"])
async def list_intents(container: Container = Depends(get_container)):
    return [
        {"intent_name": i.intent_name, "description": i.description, "examples": i.examples}
        for i in container.understanding.intents
    ]


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(db: AsyncSession = Depends(get_db)):
    from app.api.v1 import dependencies

    db_status, corpus = "connected", {}
    try:
        row = (
            await db.execute(text("SELECT (SELECT count(*) FROM tickets), (SELECT count(*) FROM kb_articles)"))
        ).one()
        corpus = {"tickets": int(row[0]), "kb_articles": int(row[1])}
    except Exception as exc:
        db_status = f"error: {exc.__class__.__name__}"
    container = dependencies._container
    llm = container.llm if container else None
    llm_status = (
        "not_configured (extractive fallback)"
        if llm is None or not llm.available
        else f"configured ({settings.LLM_MODEL})"
    )
    breaker = getattr(getattr(container, "pipeline", None), "db_breaker", None)
    circuit = breaker.state if breaker is not None else "closed"
    ok = db_status == "connected" and container is not None and circuit != "open"
    return HealthResponse(
        status="healthy" if ok else "degraded",
        database=db_status,
        llm_provider=llm_status,
        models_loaded=container is not None,
        corpus=corpus,
        version=settings.APP_VERSION,
        database_circuit=circuit,
    )


@router.get("/metrics", tags=["ops"])
async def metrics(hours: int | None = Query(None, ge=1, le=24 * 90), db: AsyncSession = Depends(get_db)):
    """Request volume, decision mix, latency percentiles, average confidence/groundedness, fallback rate."""
    return await monitoring.get_metrics(db, hours)


@router.get("/monitoring/drift", tags=["ops"])
async def drift(window_hours: int | None = Query(None, ge=1, le=24 * 30), db: AsyncSession = Depends(get_db)):
    """Recent-vs-baseline drift: unknown-intent rate, intent mix shift, nearest-neighbour similarity."""
    return await monitoring.get_drift_report(db, window_hours)
