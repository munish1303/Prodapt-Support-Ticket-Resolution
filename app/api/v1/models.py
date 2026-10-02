"""Request/response models for the public API (plan §28.2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.services.ingestion import KBArticleIn, TicketIn


class TicketResolutionRequest(BaseModel):
    complaint: str = Field(..., min_length=10, max_length=5000)
    filters: dict[str, str] | None = Field(None, description="Optional metadata filters, e.g. {'product': 'fiber'}")
    metadata: dict[str, Any] | None = None


class UnderstandingResponse(BaseModel):
    intent: str
    intent_confidence: float
    intent_candidates: list[tuple[str, float]]
    products: list[str]
    severity: str
    sentiment: str
    sentiment_score: float


class ResolutionResponse(BaseModel):
    steps: list[str]
    summary: str
    estimated_time: str | None
    generator: str


class SourceResponse(BaseModel):
    source_number: int
    type: str
    id: str
    title: str | None = None
    excerpt: str
    relevance_score: float


class ClaimValidationResponse(BaseModel):
    step: int
    claim: str
    citations: list[int]
    support_status: str
    signals: dict[str, float]


class ValidationResponse(BaseModel):
    citation_accuracy: float
    citation_coverage: float
    groundedness_score: float
    claims: list[ClaimValidationResponse]
    issues: list[str]


class ConfidenceResponse(BaseModel):
    heuristic_score: float
    interpretation: str
    components: dict[str, float]
    evidence_sufficient: bool
    note: str = (
        "Heuristic score derived from evidence signals; not a calibrated probability and not LLM self-assessment."
    )


class DecisionResponse(BaseModel):
    action: str
    reason: str


class TicketResolutionResponse(BaseModel):
    request_id: str | None
    understanding: UnderstandingResponse
    resolution: ResolutionResponse
    sources: list[SourceResponse]
    validation: ValidationResponse
    confidence: ConfidenceResponse
    decision: DecisionResponse
    flags: list[str]
    latency_ms: int
    stage_latency_ms: dict[str, int]


class IngestionRequest(BaseModel):
    tickets: list[TicketIn] = Field(default_factory=list, max_length=5000)
    kb_articles: list[KBArticleIn] = Field(default_factory=list, max_length=1000)


class IngestionResponse(BaseModel):
    tickets_inserted: int
    tickets_updated: int
    tickets_unchanged: int
    kb_inserted: int
    kb_updated: int
    kb_unchanged: int
    embeddings_generated: int
    new_categories: list[str]
    errors: list[str]


class IntentCreateRequest(BaseModel):
    intent_name: str = Field(..., pattern=r"^[a-z][a-z0-9_]{2,99}$")
    description: str = Field(..., min_length=5, max_length=500)
    examples: list[str] = Field(default_factory=list, max_length=50)


class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = Field(None, max_length=2000)


class HealthResponse(BaseModel):
    status: str
    database: str
    llm_provider: str
    models_loaded: bool
    corpus: dict[str, int]
    version: str
