"""End-to-end resolution pipeline: understand -> retrieve -> generate -> validate -> decide -> record.

Graceful degradation (plan §36.3): a failing dependency never turns into an error for the agent. Each stage has a
safe fallback and a flag in the response:

* embedding or retrieval fails (database down, timeout, open circuit) -> no LLM call, ESCALATE with the reason;
* understanding fails -> intent treated as unknown, so the decision is at most REVIEW;
* generation fails (beyond the built-in LLM -> extractive fallback) -> no draft, ESCALATE;
* validation fails -> the draft is returned unvalidated and the decision is at most REVIEW;
* the audit-log write fails -> logged, the response is still returned.

Database stages run through a circuit breaker so that, during an outage, requests are escalated immediately
instead of each waiting for a timeout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import TypeVar

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.core.resilience import OPEN, CircuitBreaker
from app.models.embeddings import EmbeddingService
from app.models.schemas import (
    ClaimCheck,
    DecisionResult,
    GenerationResult,
    QueryMetadata,
    RetrievalResult,
    ValidationResult,
)
from app.services.decision import ESCALATE, REVIEW, DecisionService
from app.services.generation import GenerationService
from app.services.retrieval import SupportsRetrieve
from app.services.understanding import UNKNOWN_INTENT, UnderstandingService
from app.services.validation import ValidationService, extract_citations
from app.utils.text import looks_like_prompt_injection, redact_pii, sha256

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions that mean "the database is unhealthy" (and count towards opening the circuit). TimeoutError covers
# asyncio.wait_for timeouts; OSError covers refused/reset connections. Other exceptions still degrade the request
# but do not trip the breaker.
DB_FAILURES: tuple[type[BaseException], ...] = (SQLAlchemyError, OSError, ConnectionError, TimeoutError)

RETRIEVAL_DOWN_REASON = "Knowledge base unavailable (retrieval failed) - escalate to a human agent"
VALIDATION_DOWN_REASON = "Steps could not be validated (validation error) - check the draft against its sources"


def fallback_metadata() -> QueryMetadata:
    """Understanding failed: treat the complaint as an unrecognised issue so a human looks at it."""
    return QueryMetadata(
        intent=UNKNOWN_INTENT,
        intent_confidence=0.0,
        products=[],
        severity="medium",
        sentiment="neutral",
        sentiment_score=0.0,
        is_unknown_intent=True,
    )


def unvalidated(steps: list[str]) -> ValidationResult:
    """Validation failed: report every step as unverified (zero groundedness) rather than guessing."""
    claims = []
    for step in steps:
        claim, cites = extract_citations(step)
        claims.append(ClaimCheck(claim, cites, [], "unverified"))
    return ValidationResult(claims, 0.0, 0.0, 0.0, ["validation unavailable"])


@dataclass
class PipelineResult:
    metadata: QueryMetadata
    retrieval: RetrievalResult
    generation: GenerationResult
    validation: ValidationResult
    decision: DecisionResult
    flags: list[str]
    latency_ms: int
    stage_ms: dict[str, int] = field(default_factory=dict)
    request_id: str | None = None


class ResolutionPipeline:
    def __init__(
        self,
        embedder: EmbeddingService,
        understanding: UnderstandingService,
        retrieval: SupportsRetrieve,
        generation: GenerationService,
        validation: ValidationService,
        decision: DecisionService,
        session_factory=None,
        db_breaker: CircuitBreaker | None = None,
        db_timeout_s: float | None = None,
    ):
        self.embedder = embedder
        self.understanding = understanding
        self.retrieval = retrieval
        self.generation = generation
        self.validation = validation
        self.decision = decision
        self.session_factory = session_factory
        self.db_breaker = db_breaker or CircuitBreaker(
            "database",
            settings.DB_CIRCUIT_FAILURE_THRESHOLD,
            settings.DB_CIRCUIT_RECOVERY_S,
            failure_types=DB_FAILURES,
            half_open_max_calls=2,  # understanding + retrieval of one request
        )
        self.db_timeout_s = settings.DB_STAGE_TIMEOUT_S if db_timeout_s is None else db_timeout_s

    async def _db_stage(self, fn: Callable[[], Awaitable[T]]) -> T:
        return await self.db_breaker.call(lambda: asyncio.wait_for(fn(), timeout=self.db_timeout_s))

    async def process(
        self, complaint: str, metadata_filters: dict | None = None, record: bool = True
    ) -> PipelineResult:
        started = time.perf_counter()
        stage: dict[str, int] = {}

        def lap(name: str, t0: float) -> float:
            now = time.perf_counter()
            stage[name] = int((now - t0) * 1000)
            return now

        flags: list[str] = []
        retrieval_down = False

        t = time.perf_counter()
        try:
            embedding = await asyncio.to_thread(self.embedder.encode, complaint)
        except Exception:
            logger.exception("embedding failed")
            embedding, retrieval_down = None, True
            flags.append("embedding_unavailable")
        t = lap("embed", t)

        if embedding is None:
            metadata, retrieval = fallback_metadata(), RetrievalResult([], "unavailable")
            flags.append("understanding_unavailable")
        else:
            # Understanding and retrieval are independent: run them concurrently.
            understood, retrieved = await asyncio.gather(
                self._db_stage(lambda: self.understanding.analyze(complaint, embedding)),
                self._db_stage(
                    lambda: self.retrieval.retrieve(
                        complaint, query_embedding=embedding, metadata_filters=metadata_filters
                    )
                ),
                return_exceptions=True,
            )
            if isinstance(understood, BaseException):
                logger.error("understanding failed: %r", understood)
                metadata = fallback_metadata()
                flags.append("understanding_unavailable")
            else:
                metadata = understood
            if isinstance(retrieved, BaseException):
                logger.error("retrieval failed: %r", retrieved)
                retrieval, retrieval_down = RetrievalResult([], "unavailable"), True
                flags.append("retrieval_unavailable")
            else:
                retrieval = retrieved
        t = lap("understand_retrieve", t)

        if retrieval_down:
            generation = GenerationResult([], "The knowledge base could not be searched.", None, generator="none")
        else:
            try:
                generation = await self.generation.generate(complaint, metadata, retrieval.documents)
            except Exception as exc:
                logger.exception("generation failed")
                generation = GenerationResult(
                    [], "No draft could be generated.", None, generator="none", error=f"generation_error: {exc}"
                )
        t = lap("generate", t)

        validation_down = False
        try:
            validation = await asyncio.to_thread(
                self.validation.validate, generation.resolution_steps, retrieval.documents
            )
        except Exception:
            logger.exception("validation failed")
            validation, validation_down = unvalidated(generation.resolution_steps), True
            flags.append("validation_unavailable")
        t = lap("validate", t)

        injection = looks_like_prompt_injection(complaint)
        decision = self.decision.decide(retrieval, validation, metadata, injection)
        if retrieval_down:
            decision = replace(decision, decision=ESCALATE, decision_reason=RETRIEVAL_DOWN_REASON)
        elif validation_down and generation.resolution_steps and decision.decision != REVIEW:
            relevant = max(retrieval.relevance_scores, default=0.0) >= settings.EVIDENCE_MIN_TOP_RELEVANCE
            if relevant:  # never auto-RESOLVE an unvalidated draft, but don't hide a usable one either
                decision = replace(decision, decision=REVIEW, decision_reason=VALIDATION_DOWN_REASON)
        lap("decide", t)

        if metadata.is_unknown_intent:
            flags.append("unknown_intent")
        if injection:
            flags.append("possible_prompt_injection")
        if generation.error:
            flags.append(
                "llm_unavailable_extractive_fallback" if generation.generator == "extractive" else "generation_error"
            )
        if any(c.support_status == "contradicted" for c in validation.claims):
            flags.append("contradiction_detected")
        if decision.heuristic_confidence < 0.4:
            flags.append("low_confidence")

        result = PipelineResult(
            metadata,
            retrieval,
            generation,
            validation,
            decision,
            flags,
            int((time.perf_counter() - started) * 1000),
            stage,
        )
        if record and self.session_factory is not None:
            # Skipped while the circuit is open and bounded by the same timeout, so an outage adds no wait here.
            # Write failures don't trip the breaker: answering depends on reads, not on the audit log.
            if self.db_breaker.state == OPEN:
                logger.warning("audit log skipped: database circuit open")
            else:
                try:
                    result.request_id = await asyncio.wait_for(
                        self._record(complaint, result), timeout=self.db_timeout_s
                    )
                except Exception:  # monitoring must never break the request path
                    logger.exception("failed to record resolution request")
        logger.info(
            "resolution",
            extra={
                "extra_fields": {
                    "decision": decision.decision,
                    "confidence": decision.heuristic_confidence,
                    "intent": metadata.intent,
                    "latency_ms": result.latency_ms,
                    "stage_ms": stage,
                    "generator": generation.generator,
                    "flags": flags,
                }
            },
        )
        return result

    async def _record(self, complaint: str, r: PipelineResult) -> str:
        async with self.session_factory() as session:
            row = (
                await session.execute(
                    text("""
                INSERT INTO resolution_requests (complaint_hash, complaint_redacted, extracted_metadata, retrieved_sources,
                    retrieval_scores, generated_resolution, citations, citation_accuracy, citation_coverage,
                    groundedness_score, validation_issues, heuristic_confidence, confidence_components, decision,
                    decision_reason, flags, generator, latency_ms, stage_latency_ms)
                VALUES (:h, :c, CAST(:meta AS jsonb), CAST(:src AS jsonb), :scores, CAST(:gen AS jsonb), CAST(:cites AS jsonb),
                    :acc, :cov, :grd, CAST(:issues AS jsonb), :conf, CAST(:comp AS jsonb), :dec, :reason,
                    CAST(:flags AS jsonb), :generator, :lat, CAST(:stage AS jsonb))
                RETURNING request_id
                """),
                    {
                        "h": sha256(complaint),
                        "c": redact_pii(complaint),
                        "meta": json.dumps(
                            {
                                "intent": r.metadata.intent,
                                "intent_confidence": r.metadata.intent_confidence,
                                "products": r.metadata.products,
                                "severity": r.metadata.severity,
                                "sentiment": r.metadata.sentiment,
                                "nearest_similarity": r.metadata.nearest_similarity,
                            }
                        ),
                        "src": json.dumps(
                            [
                                {"id": i.document.id, "type": i.document.doc_type, "relevance": round(i.relevance, 4)}
                                for i in r.retrieval.items
                            ]
                        ),
                        "scores": [round(x, 4) for x in r.retrieval.relevance_scores],
                        "gen": json.dumps({"steps": r.generation.resolution_steps, "summary": r.generation.summary}),
                        "cites": json.dumps(
                            [
                                {"claim": c.claim, "citations": c.citations, "status": c.support_status}
                                for c in r.validation.claims
                            ]
                        ),
                        "acc": r.validation.citation_accuracy,
                        "cov": r.validation.citation_coverage,
                        "grd": r.validation.groundedness_score,
                        "issues": json.dumps(r.validation.issues),
                        "conf": r.decision.heuristic_confidence,
                        "comp": json.dumps(r.decision.confidence_components),
                        "dec": r.decision.decision,
                        "reason": r.decision.decision_reason,
                        "flags": json.dumps(r.flags),
                        "generator": r.generation.generator,
                        "lat": r.latency_ms,
                        "stage": json.dumps(r.stage_ms),
                    },
                )
            ).one()
            await session.commit()
            return str(row[0])
