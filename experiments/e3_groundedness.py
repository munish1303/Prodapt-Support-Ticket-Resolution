"""Experiment 3 - Groundedness / citation validation methods (MUST RUN).

Hypothesis: combining semantic similarity + NLI + lexical overlap detects
unsupported/contradicted claims better than any single signal.

Data: data/evaluation/groundedness_eval.jsonl - (claim, source, label) triples built
from scenarios: supported claims are *paraphrases* of a step present in the source;
unsupported claims are real fix steps from a different scenario of the same intent
(hard negatives sharing vocabulary); contradicted claims directly oppose the fix.

Protocol: split by scenario into dev/test halves (no scenario in both). For each
method, thresholds are grid-searched on dev to maximise F1 for detecting
NOT-supported claims, then reported on test. Default (untuned) config thresholds
are reported too.

Usage: python experiments/e3_groundedness.py [--nli-models cross-encoder/nli-deberta-v3-small ...]
"""

from __future__ import annotations

import argparse
import itertools
from typing import Any
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.models.embeddings import get_embedding_service  # noqa: E402
from app.models.schemas import Document  # noqa: E402
from app.services.validation import GroundednessChecker, GroundednessThresholds, NLIModel, decide_support  # noqa: E402
from evaluation.data import PROCESSED_DIR, load_eval, read_jsonl, save_result  # noqa: E402
from evaluation.metrics import binary_prf  # noqa: E402

GRID = {
    "sim": [0.4, 0.5, 0.6, 0.7, 0.8],
    "entail": [0.3, 0.5, 0.7, 0.9],
    "lexical": [0.3, 0.4, 0.5, 0.6, 0.7],
    "contradiction": [0.5, 0.7, 0.9],
}


def thresholds_from(sim, entail, lexical, contradiction) -> GroundednessThresholds:
    return GroundednessThresholds(
        sim=sim,
        sim_weak=sim - 0.15,
        entail=entail,
        entail_weak=entail / 2,
        lexical=lexical,
        lexical_weak=lexical - 0.2,
        contradiction=contradiction,
    )


def evaluate(items, scores, method, t: GroundednessThresholds, weak_as_supported: bool = False) -> dict:
    preds = [decide_support(s, t, method) for s in scores]
    supported_pred = [p == "supported" or (weak_as_supported and p == "weakly_supported") for p in preds]
    y_true = [it["label"] != "supported" for it in items]  # positive class = NOT supported
    y_pred = [not sp for sp in supported_pred]
    m = binary_prf(y_true, y_pred)
    contra = [it for it in items if it["label"] == "contradicted"]
    if contra:
        hits = sum(1 for it, p in zip(items, preds) if it["label"] == "contradicted" and p == "contradicted")
        m["contradiction_recall"] = round(hits / len(contra), 4)
    false_contra = sum(1 for it, p in zip(items, preds) if it["label"] == "supported" and p == "contradicted")
    m["supported_flagged_contradicted"] = false_contra
    return m


def tune(items, scores, method):
    best, best_t = None, None
    for sim, ent, lex, con in itertools.product(GRID["sim"], GRID["entail"], GRID["lexical"], GRID["contradiction"]):
        if method == "similarity" and (ent, lex, con) != (0.5, 0.5, 0.7):
            continue
        if method == "lexical" and (sim, ent, con) != (0.6, 0.5, 0.7):
            continue
        if method == "nli" and (sim, lex) != (0.6, 0.5):
            continue
        t = thresholds_from(sim, ent, lex, con)
        m = evaluate(items, scores, method, t)
        key = (m["f1"], m["accuracy"])
        if best is None or key > best:
            best, best_t = key, t
    return best_t


CONFIGS: dict[str, dict[str, Any]] = {
    # v1: what E3 originally validated - 400-char chunk premises, max contradiction, no quote rule.
    "v1_chunk_max": {"premise_unit": "chunk", "contradiction_agg": "max", "use_quote": False},
    # v2: sentence premises, contradiction from the most similar premise, verbatim quotes => supported.
    "v2_sentence_topsim_quote": {"premise_unit": "sentence", "contradiction_agg": "top_similarity", "use_quote": True},
}


def verbatim_items() -> list[dict]:
    """Supported claims copied verbatim from a real corpus ticket (what extractive drafts - and many LLM
    drafts - actually contain). Deterministic: first resolved ticket per wave-1 scenario."""
    from scripts.datagen.scenarios import SCENARIOS

    tickets = read_jsonl(PROCESSED_DIR / "tickets.jsonl")
    out: list[dict[str, Any]] = []
    for sc in [s for s in SCENARIOS if s["wave"] == 1]:
        t = next(t for t in tickets if t["metadata"]["scenario_id"] == sc["id"] and t["metadata"]["steps_used"])
        out.append(
            {
                "id": f"VRB-{len(out) + 1:04d}",
                "claim": sc["steps"][t["metadata"]["steps_used"][0]][0],
                "source": f"Complaint: {t['complaint']}\nResolution: {t['resolution']}",
                "source_type": "ticket",
                "label": "supported",
                "kind": "verbatim",
                "scenario_id": sc["id"],
            }
        )
    return out


def subset_rates(items, preds) -> dict:
    """False-alarm rate per supported subset and detection rate per not-supported label."""
    out = {}
    for kind in ["paraphrase", "verbatim", "unsupported", "contradicted"]:
        idx = [i for i, it in enumerate(items) if it["kind"] == kind]
        if not idx:
            continue
        flagged = sum(1 for i in idx if preds[i] != "supported")
        key = "false_alarm_rate" if kind in ("paraphrase", "verbatim") else "detection_rate"
        out[kind] = {
            "n": len(idx),
            key: round(flagged / len(idx), 4),
            "flagged_contradicted": sum(1 for i in idx if preds[i] == "contradicted"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-small")
    args = parser.parse_args()

    items = load_eval("groundedness_eval")
    for it in items:
        it["kind"] = "paraphrase" if it["label"] == "supported" else it["label"]
    items += verbatim_items()
    scenarios = sorted({it["scenario_id"] for it in items})
    dev_sc = set(scenarios[::2])
    embedder = get_embedding_service()
    nli = NLIModel(args.nli_model)
    output: dict[str, Any] = {
        "nli_model": args.nli_model,
        "n_items": len(items),
        "kinds": {
            k: sum(1 for it in items if it["kind"] == k)
            for k in ["paraphrase", "verbatim", "unsupported", "contradicted"]
        },
        "dev_scenarios": len(dev_sc),
        "test_scenarios": len(scenarios) - len(dev_sc),
        "configs": {},
    }

    for cfg_name, cfg in CONFIGS.items():
        checker = GroundednessChecker(embedder, nli, **cfg)
        started = time.perf_counter()
        # raw signals for every item (NLI not skipped) so single-signal baselines are comparable
        scores = [
            checker.score(it["claim"], [Document(it["id"], it["source"])], skip_nli_if_quoted=False) for it in items
        ]
        per_item_ms = (time.perf_counter() - started) / len(items) * 1000
        dev = [(it, s) for it, s in zip(items, scores) if it["scenario_id"] in dev_sc]
        test = [(it, s) for it, s in zip(items, scores) if it["scenario_id"] not in dev_sc]
        dev_items, dev_scores = zip(*dev)
        test_items, test_scores = zip(*test)

        results: dict[str, Any] = {"score_ms_per_claim_full_nli": round(per_item_ms, 1), "methods": {}}
        for method in ["similarity", "lexical", "nli", "multi"]:
            tuned = tune(dev_items, dev_scores, method)
            test_preds = [decide_support(sc, tuned, method) for sc in test_scores]
            results["methods"][method] = {
                "tuned_thresholds": tuned.__dict__,
                "test_tuned": evaluate(test_items, test_scores, method, tuned),
                "test_tuned_weak_as_supported": (
                    evaluate(test_items, test_scores, method, tuned, True) if method == "multi" else None
                ),
                "test_subsets": subset_rates(list(test_items), test_preds),
            }
            r = results["methods"][method]["test_tuned"]
            sub = results["methods"][method]["test_subsets"]
            print(
                f"[{cfg_name}] {method:10s} F1(not-supp)={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
                f"acc={r['accuracy']:.3f} | false-alarm para={sub['paraphrase']['false_alarm_rate']:.2f} "
                f"verb={sub['verbatim']['false_alarm_rate']:.2f} | contra-recall={r.get('contradiction_recall')}"
            )
        by_kind: dict[str, list[dict[str, float]]] = {}
        for it, sc in zip(items, scores):
            by_kind.setdefault(it["kind"], []).append(sc)
        results["mean_signals_by_kind"] = {
            k: {m: round(sum(x[m] for x in ss) / len(ss), 4) for m in ss[0]} for k, ss in by_kind.items()
        }
        output["configs"][cfg_name] = results

    path = save_result("e3_groundedness", output)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
