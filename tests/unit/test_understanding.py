import numpy as np
import pytest

from app.services.understanding import (
    UNKNOWN_INTENT,
    InMemoryNeighbourIndex,
    IntentDef,
    Neighbour,
    ProductExtractor,
    UnderstandingService,
    classify_severity,
    infer_products,
    knn_vote,
    load_taxonomy_files,
)


@pytest.fixture
def taxonomy():
    return load_taxonomy_files()


def test_taxonomy_files_load(taxonomy):
    intents, products = taxonomy
    names = {i.intent_name for i in intents}
    assert {"connectivity_issue", "billing_dispute", "unknown_intent"} <= names
    assert any(p.product_name == "fiber" for p in products)


def test_product_extraction_word_boundaries(taxonomy):
    extractor = ProductExtractor(taxonomy[1])
    assert extractor.extract("My fibre ONT shows a red light and the router is fine") == ["fiber", "router"]
    assert "billing_system" in extractor.extract("I was charged twice on my bill")
    # 'sim' must not match inside 'similar'
    assert "mobile_service" not in extractor.extract("a similar problem happened before")


def test_knn_vote_majority_and_unknown():
    result = knn_vote([("a", 0.9), ("a", 0.85), ("b", 0.6)], min_confidence=0.5, min_similarity=0.4)
    assert result["intent"] == "a" and result["confidence"] > 0.7
    low_sim = knn_vote([("a", 0.3), ("a", 0.25)], min_confidence=0.5, min_similarity=0.4)
    assert low_sim["intent"] == UNKNOWN_INTENT and low_sim["is_unknown"]
    split = knn_vote([("a", 0.8), ("b", 0.8), ("c", 0.8)], min_confidence=0.5, min_similarity=0.4)
    assert split["intent"] == UNKNOWN_INTENT
    assert knn_vote([], 0.5, 0.4)["intent"] == UNKNOWN_INTENT


def test_severity_uses_neighbours_then_cues():
    neighbours = [("medium", 0.8)] * 5 + [("high", 0.7)] * 2
    assert classify_severity("my internet is slow", neighbours, 0.0, False) == "medium"
    assert classify_severity("I work from home and it is slow", neighbours, 0.0, False) == "high"
    assert classify_severity("whole street has no internet", neighbours, 0.0, False) == "critical"
    assert classify_severity("how do I set this up", [("medium", 0.9)], 0.0, False) == "low"
    assert classify_severity("no neighbours at all", [], 0.0, False) in {"medium", "high"}


def test_severity_negative_sentiment_bump(monkeypatch):
    from app.config import settings

    assert classify_severity("it is slow", [("medium", 0.9)], -0.95, False) == "medium"  # disabled by default
    monkeypatch.setattr(settings, "SEVERITY_NEGATIVE_SENTIMENT_BUMP", -0.7)
    assert classify_severity("it is slow", [("medium", 0.9)], -0.95, False) == "high"


@pytest.mark.asyncio
async def test_understanding_service_knn(embedder, taxonomy):
    intents, products = taxonomy
    corpus = [
        ("wifi keeps dropping every evening router", "connectivity_issue", "high"),
        ("wifi connection drops at night on the router", "connectivity_issue", "high"),
        ("charged twice on my bill refund", "billing_dispute", "medium"),
        ("double payment taken from bank for bill", "billing_dispute", "medium"),
    ]
    index = InMemoryNeighbourIndex(
        embedder.encode([c[0] for c in corpus]), [c[1] for c in corpus], [c[2] for c in corpus]
    )
    service = UnderstandingService(embedder, index, intents, products, sentiment=None, classifier="knn")
    meta = await service.analyze("My wifi drops every evening, the router lights look fine")
    assert meta.intent == "connectivity_issue"
    assert "router" in meta.products
    assert meta.severity in {"high", "critical"}
    assert meta.sentiment == "neutral"  # no sentiment model in this test

    # A new class becomes predictable as soon as labelled examples exist (evolving classes).
    service.set_taxonomy(
        intents + [IntentDef("roaming_issue", "Problems abroad", ["data roaming not working abroad"])], products
    )
    new_index = InMemoryNeighbourIndex(
        np.vstack(
            [index.embeddings, embedder.encode(["mobile data roaming abroad not working", "roaming data abroad fails"])]
        ),
        index.categories + ["roaming_issue", "roaming_issue"],
        index.severities + ["high", "high"],
    )
    service.index = new_index
    meta2 = await service.analyze("data roaming abroad not working on my phone")
    assert meta2.intent == "roaming_issue"


def test_infer_products_from_neighbours():
    nn = [
        Neighbour("connectivity_issue", "high", 0.8, ["router", "broadband"]),
        Neighbour("connectivity_issue", "high", 0.7, ["router", "broadband"]),
        Neighbour("connectivity_issue", "high", 0.6, ["router"]),
    ]
    assert infer_products(nn, 0.6) == ["router", "broadband"]
    assert infer_products(nn, 0.9) == ["router"]
    assert infer_products([], 0.5) == []
    assert infer_products(nn, 1.1) == []
