"""End-to-end resolution pipeline: understand -> retrieve -> generate -> validate -> decide -> record."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import text

from app.models.embeddings import EmbeddingService
from app.models.schemas import DecisionResult, GenerationResult, QueryMetadata, RetrievalResult, ValidationResult
from app.services.decision import DecisionService
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService
from app.services.understanding import UnderstandingService
from app.services.validation import ValidationService
from app.utils.text import looks_like_prompt_injection, redact_pii, sha256

logger = logging.getLogger(__name__)


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
        retrieval: RetrievalService,
        generation: GenerationService,
        validation: ValidationService,
        decision: DecisionService,
        session_factory=None,
    ):
        self.embedder = embedder
        self.understanding = understanding
        self.retrieval = retrieval
        self.generation = generation
        self.validation = validation
        self.decision = decision
        self.session_factory = session_factory

    async def process(
        self, complaint: str, metadata_filters: dict | None = None, record: bool = True
    ) -> PipelineResult:
        started = time.perf_counter()
        stage: dict[str, int] = {}

        def lap(name: str, t0: float) -> float:
            now = time.perf_counter()
            stage[name] = int((now - t0) * 1000)
            return now

        t = time.perf_counter()
        embedding = await asyncio.to_thread(self.embedder.encode, complaint)
        t = lap("embed", t)

        # Understanding and retrieval are independent: run them concurrently.
        metadata, retrieval = await asyncio.gather(
            self.understanding.analyze(complaint, embedding),
            self.retrieval.retrieve(complaint, query_embedding=embedding, metadata_filters=metadata_filters),
        )
        t = lap("understand_retrieve", t)

        generation = await self.generation.generate(complaint, metadata, retrieval.documents)
        t = lap("generate", t)

        validation = await asyncio.to_thread(self.validation.validate, generation.resolution_steps, retrieval.documents)
        t = lap("validate", t)

        injection = looks_like_prompt_injection(complaint)
        decision = self.decision.decide(retrieval, validation, metadata, injection)
        lap("decide", t)

        flags = []
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
            try:
                result.request_id = await self._record(complaint, result)
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
