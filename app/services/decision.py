"""Evidence sufficiency, evidence-derived heuristic confidence and escalation (plan §25-27).

The confidence score is a HEURISTIC computed from observable evidence signals.
It is not a calibrated probability and it is never the LLM's self-assessment.
"""

from __future__ import annotations

import numpy as np

from app.config import settings
from app.models.schemas import DecisionResult, Document, QueryMetadata, RetrievalResult, ValidationResult

RESOLVE, REVIEW, ESCALATE = "RESOLVE", "REVIEW", "ESCALATE"


def check_evidence_sufficiency(
    relevance_scores: list[float],
    validation: ValidationResult,
    sources: list[Document],
) -> tuple[bool, str]:
    """Quality-based sufficiency. No fixed minimum number of sources: one
    authoritative, relevant, grounded KB article can be enough."""
    if not relevance_scores or max(relevance_scores) < settings.EVIDENCE_MIN_TOP_RELEVANCE:
        return False, "No sufficiently relevant sources found"
    top = sorted(relevance_scores, reverse=True)[:5]
    avg = float(np.mean(top))
    if avg < settings.EVIDENCE_MIN_AVG_RELEVANCE:
        return False, f"Low retrieval relevance (top-5 mean={avg:.2f})"
    if not validation.claims:
        return False, "No resolution steps could be drawn from the sources"
    contradicted = sum(1 for c in validation.claims if c.support_status == "contradicted")
    if contradicted:
        return False, f"{contradicted} step(s) contradicted by cited sources"
    if validation.groundedness_score < settings.EVIDENCE_MIN_GROUNDEDNESS:
        return False, f"Low groundedness ({validation.groundedness_score:.2f})"
    authoritative = [
        s
        for s in sources
        if s.doc_type == "kb_article" or (s.doc_type == "ticket" and s.metadata.get("resolved", True))
    ]
    if not authoritative:
        return False, "No authoritative sources (KB articles or resolved tickets)"
    return True, "Evidence sufficient"


def compute_confidence(
    relevance_scores: list[float],
    validation: ValidationResult,
    understanding_confidence: float,
    evidence_sufficient: bool,
) -> tuple[float, dict[str, float]]:
    top = sorted(relevance_scores, reverse=True)[:5]
    retrieval_quality = min(float(np.mean(top)) / settings.CONF_RETRIEVAL_NORMALISER, 1.0) if top else 0.0
    retrieval_quality = max(retrieval_quality, 0.0)
    citation_quality = (validation.citation_accuracy + validation.citation_coverage) / 2
    evidence_quality = 0.5 * citation_quality + 0.5 * validation.groundedness_score
    understanding_quality = max(0.0, min(understanding_confidence, 1.0))
    sufficiency = 1.0 if evidence_sufficient else settings.CONF_INSUFFICIENT_CREDIT
    score = (
        settings.CONF_W_RETRIEVAL * retrieval_quality
        + settings.CONF_W_EVIDENCE * evidence_quality
        + settings.CONF_W_UNDERSTANDING * understanding_quality
        + settings.CONF_W_SUFFICIENCY * sufficiency
    )
    components = {
        "retrieval_quality": round(retrieval_quality, 4),
        "evidence_quality": round(evidence_quality, 4),
        "understanding_quality": round(understanding_quality, 4),
        "sufficiency_score": round(sufficiency, 4),
    }
    return round(score, 4), components


def interpret_confidence(score: float) -> str:
    if score >= 0.8:
        return "High - strong evidence support"
    if score >= 0.6:
        return "Moderate - adequate evidence"
    if score >= 0.4:
        return "Low - weak evidence"
    return "Very low - insufficient evidence"


def decide(
    confidence: float,
    evidence_sufficient: bool,
    sufficiency_reason: str,
    severity: str,
    intent_is_unknown: bool,
    max_relevance: float,
    injection_suspected: bool = False,
) -> tuple[str, str]:
    if not evidence_sufficient:
        return ESCALATE, f"Insufficient evidence: {sufficiency_reason}"
    if max_relevance < settings.EVIDENCE_MIN_TOP_RELEVANCE:
        return ESCALATE, "No relevant historical resolutions found"
    if intent_is_unknown:
        return REVIEW, "Unrecognised issue category - needs human assessment"
    if injection_suspected:
        return REVIEW, "Complaint contains instruction-like text - verify the draft manually"
    if settings.CRITICAL_SEVERITY_ALWAYS_REVIEW and severity == "critical":
        return REVIEW, "Critical severity - human verification required"
    if confidence >= settings.DECISION_RESOLVE_THRESHOLD:
        return RESOLVE, "High-confidence, well-grounded resolution"
    if confidence >= settings.DECISION_REVIEW_THRESHOLD:
        return REVIEW, "Moderate confidence - agent should review before use"
    return ESCALATE, f"Low confidence ({confidence:.2f})"


class DecisionService:
    def decide(
        self,
        retrieval: RetrievalResult,
        validation: ValidationResult,
        metadata: QueryMetadata,
        injection_suspected: bool = False,
    ) -> DecisionResult:
        rel = retrieval.relevance_scores
        sufficient, reason = check_evidence_sufficiency(rel, validation, retrieval.documents)
        confidence, components = compute_confidence(rel, validation, metadata.intent_confidence, sufficient)
        decision, decision_reason = decide(
            confidence,
            sufficient,
            reason,
            metadata.severity,
            metadata.is_unknown_intent,
            max(rel) if rel else 0.0,
            injection_suspected,
        )
        return DecisionResult(confidence, components, sufficient, reason, decision, decision_reason)
