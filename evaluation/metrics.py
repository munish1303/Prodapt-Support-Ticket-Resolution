"""Metric implementations shared by evaluations and experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def capped_recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """|relevant ∩ top-k| / min(k, |relevant|). Used because many queries have far
    more than k relevant tickets, which makes plain recall@k uninformative."""
    return len(set(ranked[:k]) & relevant) / min(k, len(relevant)) if relevant else 0.0


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k if k else 0.0


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], grades: dict[str, int], k: int) -> float:
    dcg = sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = (len(s) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def bootstrap_ci(xs: Sequence[float], n: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    import random

    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(mean([rng.choice(xs) for _ in xs]) for _ in range(n))
    return means[int(n * alpha / 2)], means[int(n * (1 - alpha / 2)) - 1]


def paired_bootstrap_p(a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 0) -> float:
    """One-sided p-value that mean(a) <= mean(b) under paired resampling."""
    import random

    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return 1.0
    count = 0
    for _ in range(n):
        sample = [rng.choice(diffs) for _ in diffs]
        if mean(sample) <= 0:
            count += 1
    return count / n


def binary_prf(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
