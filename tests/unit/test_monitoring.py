from app.services.monitoring import drift_alerts, intent_tvd, tvd_threshold

BASE = {
    "n": 60,
    "unknown_intent_rate": 0.0,
    "mean_nearest_similarity": 0.64,
    "intent_distribution": {
        "connectivity_issue": 0.15,
        "billing_dispute": 0.17,
        "outage_report": 0.13,
        "technical_support": 0.55,
    },
}


def test_tvd_and_size_aware_threshold():
    recent = dict(
        BASE,
        intent_distribution={
            "connectivity_issue": 0.42,
            "billing_dispute": 0.28,
            "outage_report": 0.25,
            "technical_support": 0.05,
        },
    )
    tvd = intent_tvd(recent, BASE)
    assert tvd == 0.5
    assert tvd_threshold(recent, BASE) == round(2.5 / 60**0.5, 4)
    assert any("intent mix shifted" in a for a in drift_alerts(recent, BASE, tvd))


def test_no_alert_for_noise_or_tiny_windows():
    assert drift_alerts(BASE, BASE, intent_tvd(BASE, BASE)) == []
    tiny = dict(BASE, n=10)
    assert tvd_threshold(tiny, BASE) is None
    assert drift_alerts(tiny, BASE, 0.9) == []


def test_unknown_rate_and_similarity_alerts():
    recent = dict(BASE, unknown_intent_rate=0.3, mean_nearest_similarity=0.3)
    alerts = drift_alerts(recent, BASE, 0.0)
    assert any("unknown_intent_rate" in a for a in alerts) and any("similarity" in a for a in alerts)
