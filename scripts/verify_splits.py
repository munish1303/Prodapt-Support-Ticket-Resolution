"""Verify there is no leakage between the retrieval corpus and the evaluation sets (plan §12.5).

Checks: exact id overlap, exact complaint-text overlap, template-family overlap
(eval complaints must come only from held-out symptom paraphrases), and
embedding near-duplicates (cosine >= --near-dup threshold).
Writes data/evaluation/leakage_report.json and exits non-zero on hard failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.models.embeddings import get_embedding_service  # noqa: E402

EVAL_FILES = ["understanding_eval", "retrieval_eval", "generation_eval", "test", "novel_intent_eval"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--near-dup", type=float, default=0.95)
    args = parser.parse_args()

    corpus = read_jsonl(ROOT / "data/processed/tickets.jsonl") + read_jsonl(ROOT / "data/processed/tickets_wave2.jsonl")
    corpus_ids = {t["ticket_id"] for t in corpus}
    corpus_texts = {t["complaint"] for t in corpus}
    corpus_families = {t["metadata"]["template_family"] for t in corpus}

    report, failures = {"near_dup_threshold": args.near_dup, "sets": {}}, []
    embedder = get_embedding_service()
    corpus_list = [t["complaint"] for t in corpus]
    corpus_vecs = embedder.encode(corpus_list, batch_size=64)

    all_eval_ids: dict[str, str] = {}
    for name in EVAL_FILES:
        rows = read_jsonl(ROOT / f"data/evaluation/{name}.jsonl")
        ids = [r["id"] for r in rows]
        for i in ids:
            if i in all_eval_ids:
                failures.append(f"{i} appears in both {all_eval_ids[i]} and {name}")
            all_eval_ids[i] = name
        id_overlap = len(set(ids) & corpus_ids)
        text_overlap = sum(1 for r in rows if r["complaint"] in corpus_texts)
        family_overlap = sorted({r["template_family"] for r in rows} & corpus_families)
        vecs = embedder.encode([r["complaint"] for r in rows], batch_size=64)
        max_sims = (vecs @ corpus_vecs.T).max(axis=1)
        near = int((max_sims >= args.near_dup).sum())
        report["sets"][name] = {
            "n": len(rows),
            "id_overlap": id_overlap,
            "exact_text_overlap": text_overlap,
            "template_family_overlap": len(family_overlap),
            "near_duplicates": near,
            "max_sim_to_corpus": {
                "mean": round(float(max_sims.mean()), 4),
                "p50": round(float(np.percentile(max_sims, 50)), 4),
                "p95": round(float(np.percentile(max_sims, 95)), 4),
                "max": round(float(max_sims.max()), 4),
            },
        }
        if id_overlap or text_overlap or family_overlap:
            failures.append(f"{name}: id={id_overlap} text={text_overlap} family={family_overlap[:3]}")
        print(
            f"{name:20s} n={len(rows):4d} id_overlap={id_overlap} text_overlap={text_overlap} "
            f"family_overlap={len(family_overlap)} near_dup(>={args.near_dup})={near} "
            f"max_sim mean={max_sims.mean():.3f} p95={np.percentile(max_sims, 95):.3f}"
        )

    report["failures"] = failures
    out = ROOT / "data/evaluation/leakage_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    if failures:
        print("LEAKAGE CHECK FAILED:", *failures, sep="\n  ")
        sys.exit(1)
    print("No exact, id or template-family leakage detected.")


if __name__ == "__main__":
    main()
