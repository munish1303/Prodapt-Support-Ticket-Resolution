"""Analyse the filled human-evaluation workbook (data/evaluation/human_eval_sheet.xlsx).

Reports, over rated rows only:
  * mean human ratings with 95% bootstrap CIs; share of drafts safe to use as-is
  * human vs system decision: agreement, Cohen's kappa, confusion matrix; precision of system RESOLVE
  * validity of automatic signals (Spearman rank correlation):
      human correctness vs LLM-judge correctness   - can the LLM judge be trusted?
      human correctness vs heuristic confidence    - does confidence track quality?
      human completeness vs auto reference-step recall - is the automatic metric meaningful?
  * AUROC of heuristic confidence for predicting "safe to use"
  * RESOLVE-threshold sweep: for each threshold, share of auto-resolved drafts a human called safe (precision)
    and share of drafts auto-resolved (coverage). This is the input for Experiment 5 (threshold tuning).

Usage: python evaluation/human_eval_analysis.py [--sheet path.xlsx] [--out experiments/results/human_eval.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openpyxl import load_workbook  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score  # noqa: E402

from evaluation.data import EVAL_DIR, RESULTS_DIR  # noqa: E402
from evaluation.metrics import bootstrap_ci, mean  # noqa: E402

FIRST, LAST = 3, 52
DECISIONS = ["RESOLVE", "REVIEW", "ESCALATE"]


def load(sheet: Path) -> list[dict]:
    wb = load_workbook(sheet, data_only=True)
    rat, sysw = wb["Ratings"], wb["System"]
    rows = []
    for r in range(FIRST, LAST + 1):
        corr = rat[f"H{r}"].value
        if corr in (None, ""):
            continue
        rows.append(
            {
                "sample": rat[f"A{r}"].value,
                "relevance": _num(rat[f"F{r}"].value),
                "completeness": _num(rat[f"G{r}"].value),
                "correctness": _num(corr),
                "safe": (rat[f"I{r}"].value or "").strip().lower() == "yes" if rat[f"I{r}"].value else None,
                "human_decision": (rat[f"J{r}"].value or "").strip().upper() or None,
                "notes": rat[f"K{r}"].value,
                "case_id": sysw[f"B{r}"].value,
                "system_decision": sysw[f"E{r}"].value,
                "confidence": _num(sysw[f"F{r}"].value),
                "auto_recall": _num(sysw[f"H{r}"].value),
                "judge_correctness": _num(sysw[f"L{r}"].value),
                "declined": sysw[f"M{r}"].value == "Yes",
            }
        )
    return rows


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _spearman(xs, ys) -> dict:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 5:
        return {"n": len(pairs), "rho": None, "p": None}
    rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return {"n": len(pairs), "rho": round(float(rho), 4), "p": round(float(p), 4)}


def _ci(xs) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "mean": None, "ci95": None}
    lo, hi = bootstrap_ci(xs)
    return {"n": len(xs), "mean": round(mean(xs), 3), "ci95": [round(lo, 3), round(hi, 3)]}


def analyse(rows: list[dict]) -> dict:
    out: dict = {"n_rated": len(rows)}
    out["ratings"] = {k: _ci([r[k] for r in rows]) for k in ("relevance", "completeness", "correctness")}
    safe = [r for r in rows if r["safe"] is not None]
    out["share_safe"] = round(sum(r["safe"] for r in safe) / len(safe), 4) if safe else None

    dec = [r for r in rows if r["human_decision"] in DECISIONS and r["system_decision"] in DECISIONS]
    if dec:
        h, s = [r["human_decision"] for r in dec], [r["system_decision"] for r in dec]
        out["decision"] = {
            "n": len(dec),
            "agreement": round(sum(a == b for a, b in zip(h, s)) / len(dec), 4),
            "cohen_kappa": round(float(cohen_kappa_score(h, s, labels=DECISIONS)), 4) if len(set(h + s)) > 1 else None,
            "confusion_rows_human_cols_system": {
                "labels": DECISIONS,
                "matrix": confusion_matrix(h, s, labels=DECISIONS).tolist(),
            },
        }
    auto = [r for r in safe if r["system_decision"] == "RESOLVE"]
    out["precision_of_system_resolve"] = round(sum(r["safe"] for r in auto) / len(auto), 4) if auto else None
    out["correctness_by_system_decision"] = {
        d: _ci([r["correctness"] for r in rows if r["system_decision"] == d]) for d in DECISIONS
    }
    out["validity"] = {
        "human_correctness_vs_llm_judge_correctness": _spearman(
            [r["correctness"] for r in rows], [r["judge_correctness"] for r in rows]
        ),
        "human_correctness_vs_heuristic_confidence": _spearman(
            [r["correctness"] for r in rows], [r["confidence"] for r in rows]
        ),
        "human_completeness_vs_auto_reference_recall": _spearman(
            [r["completeness"] for r in rows], [r["auto_recall"] for r in rows]
        ),
    }
    labels = [r["safe"] for r in safe]
    if safe and 0 < sum(labels) < len(labels):
        out["confidence_auroc_for_safe"] = round(float(roc_auc_score(labels, [r["confidence"] for r in safe])), 4)
    sweep = []
    for t in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        chosen = [r for r in safe if r["confidence"] is not None and r["confidence"] >= t]
        sweep.append(
            {
                "threshold": t,
                "coverage": round(len(chosen) / len(safe), 4) if safe else None,
                "precision_safe": round(sum(r["safe"] for r in chosen) / len(chosen), 4) if chosen else None,
            }
        )
    out["resolve_threshold_sweep"] = sweep
    out["notes"] = [{"sample": r["sample"], "case_id": r["case_id"], "note": r["notes"]} for r in rows if r["notes"]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", default=str(EVAL_DIR / "human_eval_sheet.xlsx"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "human_eval.json"))
    args = parser.parse_args()
    rows = load(Path(args.sheet))
    if not rows:
        raise SystemExit("no rated rows yet: fill the yellow cells on the Ratings sheet first")
    result = analyse(rows)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "notes"}, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
