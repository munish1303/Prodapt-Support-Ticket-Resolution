"""Carry scale-dependent thresholds over to a new embedding model by matching score distributions.

Some thresholds have a tuning procedure of their own and are re-tuned directly on the dev split (groundedness: E3;
unknown intent: the understanding sweep; RRF weights: E1). The ones below were set on the old model's cosine scale
without a dedicated procedure, so they are translated: a threshold keeps its *position* in the score distribution.

  EVIDENCE_MIN_TOP_RELEVANCE  <- top1_relevance         (best retrieved source)
  EVIDENCE_MIN_AVG_RELEVANCE  <- top5_mean_relevance    (mean of the top-5 sources)
  CONF_RETRIEVAL_NORMALISER   <- top5_mean_relevance
  DRIFT_LOW_SIMILARITY_ALERT  <- nearest_similarity     (nearest labelled ticket)

Method: pool the dev and novel-class complaints (experiments/embedding_scale_<old>.json and _<new>.json, recorded by
experiments/embedding_scale_snapshot.py on the same complaints). If the old threshold lies inside the old
distribution (2nd-98th percentile), use the new model's value at the same percentile; otherwise extrapolate with a
straight line fitted through the paired percentiles (Q-Q line). Dev/novel complaints only: no evaluation data.

Usage: python experiments/recalibrate_thresholds.py --old minilm --new mpnet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.data import RESULTS_DIR, save_result  # noqa: E402

MAPPING = {
    "EVIDENCE_MIN_TOP_RELEVANCE": "top1_relevance",
    "EVIDENCE_MIN_AVG_RELEVANCE": "top5_mean_relevance",
    "CONF_RETRIEVAL_NORMALISER": "top5_mean_relevance",
    "DRIFT_LOW_SIMILARITY_ALERT": "nearest_similarity",
}


def pooled(snapshot: dict, stat: str) -> np.ndarray:
    return np.array([v for split in snapshot["values"].values() for v in split[stat]], dtype=float)


def translate(threshold: float, old: np.ndarray, new: np.ndarray) -> dict:
    pct = float((old <= threshold).mean())
    ps = np.arange(1, 100)
    q_old, q_new = np.percentile(old, ps), np.percentile(new, ps)
    slope, intercept = np.polyfit(q_old, q_new, 1)
    by_line = float(slope * threshold + intercept)
    by_pct = float(np.percentile(new, pct * 100)) if 0.02 <= pct <= 0.98 else None
    chosen = by_pct if by_pct is not None else by_line
    return {
        "old_value": threshold,
        "old_percentile": round(pct, 4),
        "new_by_percentile": None if by_pct is None else round(by_pct, 4),
        "new_by_qq_line": round(by_line, 4),
        "qq_line": {"slope": round(float(slope), 4), "intercept": round(float(intercept), 4)},
        "method": "percentile" if by_pct is not None else "qq_line (outside observed range)",
        "new_value": round(chosen, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="minilm")
    parser.add_argument("--new", default="mpnet")
    parser.add_argument(
        "--values",
        default=json.dumps(
            {
                "EVIDENCE_MIN_TOP_RELEVANCE": 0.35,
                "EVIDENCE_MIN_AVG_RELEVANCE": 0.40,
                "CONF_RETRIEVAL_NORMALISER": 0.80,
                "DRIFT_LOW_SIMILARITY_ALERT": 0.45,
            }
        ),
        help="old-model threshold values (JSON)",
    )
    args = parser.parse_args()
    old_snap = json.loads((RESULTS_DIR / f"embedding_scale_{args.old}.json").read_text(encoding="utf-8"))
    new_snap = json.loads((RESULTS_DIR / f"embedding_scale_{args.new}.json").read_text(encoding="utf-8"))
    values = json.loads(args.values)
    out = {}
    for name, stat in MAPPING.items():
        old, new = pooled(old_snap, stat), pooled(new_snap, stat)
        out[name] = {"statistic": stat, **translate(values[name], old, new)}
        r = out[name]
        print(
            f"{name:28s} {r['old_value']:.2f} (old p{r['old_percentile'] * 100:.0f}) -> {r['new_value']:.2f} [{r['method']}]"
        )
    summary = {
        stat: {
            "old_median": round(float(np.median(pooled(old_snap, stat))), 4),
            "new_median": round(float(np.median(pooled(new_snap, stat))), 4),
        }
        for stat in sorted(set(MAPPING.values()))
    }
    payload = {
        "old_model": old_snap["embedding_model"],
        "new_model": new_snap["embedding_model"],
        "n_complaints": old_snap["n"],
        "medians": summary,
        "thresholds": out,
    }
    print(f"saved {save_result('threshold_recalibration', payload)}")


if __name__ == "__main__":
    main()
