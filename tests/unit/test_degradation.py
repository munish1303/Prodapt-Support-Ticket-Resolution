"""Graceful degradation: a failing dependency yields a safe, flagged answer instead of an error (plan §36.3)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.dependencies import Container, set_container
from app.api.v1.routes import router
from app.core.database import get_db
from app.core.resilience import CircuitBreaker
from app.models.schemas import RetrievalResult, RetrievedDocument
from app.services.decision import ESCALATE, RESOLVE, REVIEW, DecisionService
from app.services.generation import ExtractiveGenerator, GenerationService
from app.services.pipeline import DB_FAILURES, RETRIEVAL_DOWN_REASON, VALIDATION_DOWN_REASON, ResolutionPipeline
from app.services.understanding import InMemoryNeighbourIndex, UnderstandingService, load_taxonomy_files
from app.services.validation import GroundednessChecker, GroundednessThresholds, ValidationService

COMPLAINT = "My wifi drops every evening around 8pm and I have restarted the router twice."


class Retrieval:
    def __init__(self, docs, fail: BaseException | None = None, delay: float = 0.0):
        self.docs, self.fail, self.delay, self.calls = docs, fail, delay, 0

    async def retrieve(self, query_text, top_k=None, query_embedding=None, metadata_filters=None):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        return RetrievalResult([RetrievedDocument(d, 1.0 / (i + 1), 0.8) for i, d in enumerate(self.docs)], "fake")


class FailingUnderstanding:
    def __init__(self):
        self.calls = 0

    async def analyze(self, complaint, embedding):
        self.calls += 1
        raise OSError("connection refused")


class SpyGeneration(GenerationService):
    def __init__(self, crash: bool = False):
        super().__init__(None, ExtractiveGenerator())
        self.calls, self.crash = 0, crash

    async def generate(self, complaint, metadata, documents):
        self.calls += 1
        if self.crash:
            raise RuntimeError("unexpected generator bug")
        return await super().generate(complaint, metadata, documents)


class FailingValidation:
    def validate(self, steps, sources):
        raise RuntimeError("NLI model crashed")


class FailingEmbedder:
    def encode(self, texts, **kw):
        raise RuntimeError("embedding model crashed")


@pytest.fixture
def parts(embedder, fake_nli, sources):
    intents, products = load_taxonomy_files()
    corpus = [
        ("wifi drops every evening router channel", "connectivity_issue", "medium"),
        ("wireless connection drops at night", "connectivity_issue", "medium"),
        ("charged twice on bill", "billing_dispute", "medium"),
    ]
    index = InMemoryNeighbourIndex(
        embedder.encode([c[0] for c in corpus]), [c[1] for c in corpus], [c[2] for c in corpus]
    )
    thresholds = GroundednessThresholds(
        sim=0.5, sim_weak=0.3, entail=0.5, entail_weak=0.25, lexical=0.5, lexical_weak=0.3, contradiction=0.7
    )
    return {
        "embedder": embedder,
        "understanding": UnderstandingService(embedder, index, intents, products, sentiment=None),
        "retrieval": Retrieval(sources),
        "generation": SpyGeneration(),
        "validation": ValidationService(GroundednessChecker(embedder, fake_nli, thresholds)),
        "sources": sources,
    }


def build(parts, **overrides) -> ResolutionPipeline:
    p = {**parts, **overrides}
    return ResolutionPipeline(
        p["embedder"],
        p["understanding"],
        p["retrieval"],
        p["generation"],
        p["validation"],
        DecisionService(),
        db_breaker=p.get("breaker"),
        db_timeout_s=p.get("timeout", 5.0),
    )


async def test_healthy_path_is_unchanged(parts):
    r = await build(parts).process(COMPLAINT, record=False)
    assert r.generation.resolution_steps and r.decision.decision in {RESOLVE, REVIEW}
    assert not {"retrieval_unavailable", "understanding_unavailable", "validation_unavailable"} & set(r.flags)


async def test_retrieval_failure_escalates_without_calling_the_llm(parts):
    gen = SpyGeneration()
    r = await build(parts, retrieval=Retrieval(parts["sources"], fail=OSError("refused")), generation=gen).process(
        COMPLAINT, record=False
    )
    assert r.decision.decision == ESCALATE and r.decision.decision_reason == RETRIEVAL_DOWN_REASON
    assert "retrieval_unavailable" in r.flags and gen.calls == 0
    assert r.generation.generator == "none" and not r.generation.resolution_steps


async def test_slow_database_counts_as_a_failure(parts):
    slow = Retrieval(parts["sources"], delay=1.0)
    r = await build(parts, retrieval=slow, timeout=0.05).process(COMPLAINT, record=False)
    assert "retrieval_unavailable" in r.flags and r.decision.decision == ESCALATE


async def test_open_circuit_skips_the_database_entirely(parts):
    breaker = CircuitBreaker("database", 2, 60, failure_types=DB_FAILURES, half_open_max_calls=2)
    down = Retrieval(parts["sources"], fail=OSError("refused"))
    understanding = FailingUnderstanding()
    pipeline = build(parts, retrieval=down, understanding=understanding, breaker=breaker)
    await pipeline.process(COMPLAINT, record=False)  # 2 failures (understanding + retrieval) -> open
    assert breaker.state == "open"
    r = await pipeline.process(COMPLAINT, record=False)
    assert down.calls == 1 and understanding.calls == 1  # second request never reached the database
    assert r.decision.decision == ESCALATE and {"retrieval_unavailable", "understanding_unavailable"} <= set(r.flags)


async def test_understanding_failure_degrades_to_unknown_intent(parts):
    r = await build(parts, understanding=FailingUnderstanding()).process(COMPLAINT, record=False)
    assert {"understanding_unavailable", "unknown_intent"} <= set(r.flags)
    assert r.metadata.intent == "unknown_intent" and r.generation.resolution_steps
    assert r.decision.decision != RESOLVE


async def test_validation_failure_returns_the_draft_unvalidated_for_review(parts):
    r = await build(parts, validation=FailingValidation()).process(COMPLAINT, record=False)
    assert "validation_unavailable" in r.flags and r.generation.resolution_steps
    assert {c.support_status for c in r.validation.claims} == {"unverified"}
    assert r.decision.decision == REVIEW and r.decision.decision_reason == VALIDATION_DOWN_REASON


async def test_generation_crash_escalates_with_flag(parts):
    r = await build(parts, generation=SpyGeneration(crash=True)).process(COMPLAINT, record=False)
    assert "generation_error" in r.flags and not r.generation.resolution_steps
    assert r.decision.decision == ESCALATE


async def test_embedding_failure_escalates(parts):
    r = await build(parts, embedder=FailingEmbedder()).process(COMPLAINT, record=False)
    assert {"embedding_unavailable", "understanding_unavailable"} <= set(r.flags)
    assert r.decision.decision == ESCALATE


def test_api_returns_200_with_flags_and_health_reports_the_circuit(parts):
    # Database outage: both database stages fail (a single healthy database call would close the circuit again).
    breaker = CircuitBreaker("database", 1, 60, failure_types=DB_FAILURES, half_open_max_calls=2)
    pipeline = build(
        parts,
        retrieval=Retrieval(parts["sources"], fail=OSError("refused")),
        understanding=FailingUnderstanding(),
        breaker=breaker,
    )
    set_container(Container(parts["embedder"], None, parts["understanding"], pipeline, None, None))

    class DownSession:
        async def execute(self, *a, **kw):
            raise OSError("connection refused")

    async def down_db():
        yield DownSession()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = down_db
    try:
        client = TestClient(app)
        resp = client.post("/api/v1/tickets/resolve", json={"complaint": COMPLAINT})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["decision"]["action"] == ESCALATE and "retrieval_unavailable" in body["flags"]
        health = client.get("/api/v1/health").json()
        assert health["status"] == "degraded" and health["database_circuit"] == "open"
        assert health["database"].startswith("error")
    finally:
        set_container(None)


class AuditLog:
    """Session factory stand-in whose writes fail; counts how often a write was attempted."""

    def __init__(self) -> None:
        self.attempts = 0

    def __call__(self):
        log = self

        class Session:
            async def __aenter__(self):
                log.attempts += 1
                return self

            async def __aexit__(self, *exc):
                return False

            async def execute(self, *a, **kw):
                raise OSError("disk full")

        return Session()


async def test_audit_write_failure_does_not_trip_the_breaker(parts):
    breaker = CircuitBreaker("database", 1, 60, failure_types=DB_FAILURES, half_open_max_calls=2)
    audit = AuditLog()
    pipeline = build(parts, breaker=breaker)
    pipeline.session_factory = audit
    r = await pipeline.process(COMPLAINT)
    assert audit.attempts == 1 and r.request_id is None  # write failed, answer still returned
    assert breaker.state == "closed" and r.decision.decision in {RESOLVE, REVIEW}


async def test_audit_write_is_skipped_while_the_circuit_is_open(parts):
    breaker = CircuitBreaker("database", 2, 60, failure_types=DB_FAILURES, half_open_max_calls=2)
    audit = AuditLog()
    pipeline = build(
        parts,
        retrieval=Retrieval(parts["sources"], fail=OSError("refused")),
        understanding=FailingUnderstanding(),
        breaker=breaker,
    )
    pipeline.session_factory = audit
    await pipeline.process(COMPLAINT)
    await pipeline.process(COMPLAINT)
    assert breaker.state == "open" and audit.attempts == 0
