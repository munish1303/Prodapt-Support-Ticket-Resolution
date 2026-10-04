import pytest

from app.config import settings
from app.models.schemas import ClaimCheck, Document, QueryMetadata, RetrievalResult, RetrievedDocument, ValidationResult
from app.services.decision import (
    ESCALATE,
    RESOLVE,
    REVIEW,
    DecisionService,
    check_evidence_sufficiency,
    compute_confidence,
    decide,
    interpret_confidence,
)


def vr(statuses, acc=1.0, cov=1.0, grd=None):
    claims = [ClaimCheck(f"c{i}", [1], [], s) for i, s in enumerate(statuses)]
    if grd is None:
        grd = sum(1 for s in statuses if s == "supported") / len(statuses) if statuses else 0.0
    return ValidationResult(claims, acc, cov, grd)


KB = [Document("KB-1", "kb text", {"type": "kb_article"})]


def test_single_authoritative_source_can_be_sufficient():
    ok, reason = check_evidence_sufficiency([0.8], vr(["supported", "supported"]), KB)
    assert ok, reason


@pytest.mark.parametrize(
    "scores,validation,expected_reason",
    [
        ([0.2, 0.1], vr(["supported"]), "relevant"),
        # top source just above the relevance floor, but a weak top-5 mean (values follow the configured floors)
        ([settings.EVIDENCE_MIN_TOP_RELEVANCE + 0.01, 0.2, 0.2, 0.2, 0.2], vr(["supported"]), "relevance"),
        ([0.8], vr([]), "No resolution steps"),
        ([0.8], vr(["supported", "contradicted"]), "contradicted"),
        ([0.8], vr(["unsupported", "unsupported", "supported"]), "groundedness"),
    ],
)
def test_insufficient_evidence(scores, validation, expected_reason):
    ok, reason = check_evidence_sufficiency(scores, validation, KB)
    assert not ok and expected_reason.lower() in reason.lower()


def test_unresolved_tickets_are_not_authoritative():
    docs = [Document("T", "x", {"type": "ticket", "resolved": False})]
    ok, reason = check_evidence_sufficiency([0.8], vr(["supported"]), docs)
    assert not ok and "authoritative" in reason


def test_confidence_components_and_range():
    score, comp = compute_confidence([0.8, 0.8], vr(["supported"]), 0.9, True)
    assert 0.0 <= score <= 1.0
    assert comp["retrieval_quality"] == 1.0 and comp["sufficiency_score"] == 1.0
    low, _ = compute_confidence([0.2], vr(["unsupported"], acc=0.5, cov=0.5), 0.2, False)
    assert low < score


def test_decision_rules():
    assert decide(0.9, False, "x", "low", False, 0.9)[0] == ESCALATE
    assert decide(0.9, True, "", "low", True, 0.9)[0] == REVIEW
    assert decide(0.9, True, "", "critical", False, 0.9)[0] == REVIEW
    assert decide(0.9, True, "", "low", False, 0.9, injection_suspected=True)[0] == REVIEW
    assert decide(0.9, True, "", "low", False, 0.9)[0] == RESOLVE
    assert decide(0.6, True, "", "low", False, 0.9)[0] == REVIEW
    assert decide(0.3, True, "", "low", False, 0.9)[0] == ESCALATE


def test_interpretation_buckets():
    assert interpret_confidence(0.85).startswith("High")
    assert interpret_confidence(0.1).startswith("Very low")


def test_decision_service_end_to_end():
    retrieval = RetrievalResult([RetrievedDocument(KB[0], 0.03, 0.85)], "hybrid")
    meta = QueryMetadata("connectivity_issue", 0.95, ["router"], "medium", "negative", -0.5)
    result = DecisionService().decide(retrieval, vr(["supported", "supported"]), meta)
    assert result.evidence_sufficient and result.decision == RESOLVE
