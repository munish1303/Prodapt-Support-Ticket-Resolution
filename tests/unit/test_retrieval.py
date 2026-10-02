from app.services.retrieval import HybridRetriever, build_or_tsquery, reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], k=60)
    ids = [d for d, _ in fused]
    assert ids[0] in {"a", "b"} and set(ids) == {"a", "b", "c", "d"}
    # 'b' (ranks 2 and 1) beats 'c' (rank 3 only) and 'd' (rank 2 only)
    assert ids.index("b") < ids.index("c") and ids.index("b") < ids.index("d")


def test_rrf_weights():
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], weights=[0.8, 0.2], k=60))
    assert fused["a"] > fused["b"]
    assert abs(fused["a"] - 0.8 / 61) < 1e-9


def test_or_tsquery_dedupes_and_drops_stopwords():
    q = build_or_tsquery("The router router is dropping and the WiFi is slow")
    terms = q.split(" | ")
    assert terms == ["router", "dropping", "wifi", "slow"]
    assert build_or_tsquery("the and of") == ""


def _ranked():
    tickets = [("tickets", f"T{i}", 1.0 - i * 0.01, i, None) for i in range(10)]
    kb = [
        ("kb_articles", "K1", 0.5, 11, None),
        ("kb_articles", "K2", 0.4, 12, None),
        ("kb_articles", "K3", 0.3, 13, None),
    ]
    return tickets + kb  # already ranked


def test_kb_slots_are_guaranteed():
    chosen = HybridRetriever._apply_kb_slots(_ranked(), top_k=5, kb_min_slots=2)
    ids = [r[1] for r in chosen]
    assert len(ids) == 5
    assert ids == ["T0", "T1", "T2", "K1", "K2"]


def test_no_kb_slots_keeps_ranking():
    assert [r[1] for r in HybridRetriever._apply_kb_slots(_ranked(), top_k=3, kb_min_slots=0)] == ["T0", "T1", "T2"]


def test_kb_already_present_is_not_duplicated():
    ranked = [("kb_articles", "K1", 0.9, 1, None)] + [("tickets", f"T{i}", 0.8, i, None) for i in range(5)]
    chosen = HybridRetriever._apply_kb_slots(ranked, top_k=3, kb_min_slots=1)
    assert [r[1] for r in chosen] == ["K1", "T0", "T1"]
