import pytest

from app.services.validation import (
    GroundednessChecker,
    GroundednessThresholds,
    ValidationService,
    decide_support,
    extract_citations,
    lexical_overlap,
)


def test_extract_citations_formats():
    assert extract_citations("Do X [1][3]") == ("Do X", [1, 3])
    assert extract_citations("Do Y [2, 4] then [2]") == ("Do Y  then", [2, 4])
    assert extract_citations("No cites") == ("No cites", [])


def test_lexical_overlap():
    assert lexical_overlap("restart the router", "Please restart your router now") == 1.0
    assert lexical_overlap("replace the SIM", "restart the router") == 0.0


T = GroundednessThresholds(
    sim=0.6, sim_weak=0.4, entail=0.5, entail_weak=0.25, lexical=0.5, lexical_weak=0.3, contradiction=0.7
)


@pytest.mark.parametrize(
    "scores,expected",
    [
        ({"semantic": 0.7, "entailment": 0.9, "contradiction": 0.0, "lexical": 0.2}, "supported"),
        ({"semantic": 0.8, "entailment": 0.1, "contradiction": 0.1, "lexical": 0.7}, "supported"),
        ({"semantic": 0.5, "entailment": 0.3, "contradiction": 0.1, "lexical": 0.1}, "weakly_supported"),
        ({"semantic": 0.3, "entailment": 0.1, "contradiction": 0.1, "lexical": 0.1}, "unsupported"),
        ({"semantic": 0.8, "entailment": 0.05, "contradiction": 0.95, "lexical": 0.8}, "contradicted"),
        # high similarity alone (the plan's anti-pattern) is NOT enough
        ({"semantic": 0.9, "entailment": 0.05, "contradiction": 0.2, "lexical": 0.2}, "unsupported"),
    ],
)
def test_decide_support_multi(scores, expected):
    assert decide_support(scores, T) == expected


def test_decide_support_single_method_baselines():
    s = {"semantic": 0.9, "entailment": 0.05, "contradiction": 0.95, "lexical": 0.9}
    assert decide_support(s, T, "similarity") == "supported"
    assert decide_support(s, T, "lexical") == "supported"
    assert decide_support(s, T, "nli") == "contradicted"


def test_validation_service(embedder, fake_nli, sources):
    service = ValidationService(GroundednessChecker(embedder, fake_nli, T))
    steps = [
        "Run the router's channel scan and move the 2.4 GHz radio to the least congested channel [1][2]",
        "Enable the 5 GHz band and connect nearby devices to it [3]",
        "Buy a new laptop [2]",
        "Call the president [9]",
        "Restart everything",
    ]
    result = service.validate(steps, sources)
    statuses = [c.support_status for c in result.claims]
    assert statuses[0] == "supported" and statuses[1] == "supported"
    assert statuses[2] == "unsupported"
    assert statuses[3] == "uncited" and statuses[4] == "uncited"
    assert result.citation_accuracy == pytest.approx(1 - 1 / 5)  # [9] is invalid out of 5 citations
    assert result.citation_coverage == pytest.approx(3 / 5)
    assert result.groundedness_score == pytest.approx(2 / 5)
    assert any("Invalid citation" in i for i in result.issues)


def test_validation_empty_steps(embedder, fake_nli, sources):
    result = ValidationService(GroundednessChecker(embedder, fake_nli, T)).validate([], sources)
    assert result.groundedness_score == 0.0 and result.citation_coverage == 0.0


def test_verbatim_quote_is_supported_without_nli(embedder, sources):
    class ExplodingNLI:
        def predict(self, pairs):
            raise AssertionError("NLI must be skipped for verbatim quotes")

    checker = GroundednessChecker(embedder, ExplodingNLI(), T, use_quote=True)
    status, scores = checker.check_claim("Update the router firmware to the latest version", [sources[1]])
    assert status == "supported" and scores["quoted"] == 1.0


def test_negated_claim_is_not_treated_as_quote(embedder, fake_nli, sources):
    checker = GroundednessChecker(embedder, fake_nli, T, use_quote=True)
    status, scores = checker.check_claim("Do not update the router firmware to the latest version", [sources[1]])
    assert scores["quoted"] == 0.0 and status != "supported"


def test_contradiction_aggregation_modes(embedder, sources):
    class ScriptedNLI:
        """Contradiction 0.95 only for the premise that is NOT about the claim."""

        def predict(self, pairs):
            return [
                {"entailment": 0.0, "contradiction": 0.95 if "firmware" in p.lower() else 0.02, "neutral": 0.05}
                for p, _ in pairs
            ]

    claim = "Enable the 5 GHz band for devices nearby"
    top = GroundednessChecker(embedder, ScriptedNLI(), T, contradiction_agg="top_similarity", use_quote=False)
    mx = GroundednessChecker(embedder, ScriptedNLI(), T, contradiction_agg="max", use_quote=False)
    assert top.score(claim, [sources[1]])["contradiction"] < 0.5  # most similar premise is the 5 GHz step
    assert mx.score(claim, [sources[1]])["contradiction"] == 0.95  # v1 picks up an unrelated premise
