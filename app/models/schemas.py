"""Internal data structures passed between services (not API models)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_type(self) -> str:
        return self.metadata.get("type", "unknown")


@dataclass
class QueryMetadata:
    intent: str
    intent_confidence: float
    products: list[str]
    severity: str
    sentiment: str
    sentiment_score: float
    intent_candidates: list[tuple[str, float]] = field(default_factory=list)
    nearest_similarity: float | None = None
    is_unknown_intent: bool = False


@dataclass
class RetrievedDocument:
    document: Document
    fused_score: float  # rank-fusion (or reranker) score; only meaningful for ordering
    relevance: float  # cosine similarity query↔doc in [−1, 1]; used for evidence signals
    semantic_rank: int | None = None
    lexical_rank: int | None = None


@dataclass
class RetrievalResult:
    items: list[RetrievedDocument]
    method: str

    @property
    def documents(self) -> list[Document]:
        return [i.document for i in self.items]

    @property
    def relevance_scores(self) -> list[float]:
        return [i.relevance for i in self.items]


@dataclass
class GenerationResult:
    resolution_steps: list[str]
    summary: str
    estimated_time: str | None
    generator: str
    raw_text: str = ""
    error: str | None = None


@dataclass
class ClaimCheck:
    claim: str
    citations: list[int]
    invalid_citations: list[int]
    support_status: str  # supported | weakly_supported | unsupported | contradicted | uncited
    methods: dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationResult:
    claims: list[ClaimCheck]
    citation_accuracy: float
    citation_coverage: float
    groundedness_score: float
    issues: list[str] = field(default_factory=list)


@dataclass
class DecisionResult:
    heuristic_confidence: float
    confidence_components: dict[str, float]
    evidence_sufficient: bool
    sufficiency_reason: str
    decision: str
    decision_reason: str
