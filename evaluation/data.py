"""Loaders for evaluation datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "evaluation"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "experiments" / "results"


def read_jsonl(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval(name: str) -> list[dict]:
    return read_jsonl(EVAL_DIR / f"{name}.jsonl")


def load_qrels(path: Path | None = None) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for line in (path or EVAL_DIR / "retrieval_qrels.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, _, doc, grade = line.split()
            qrels[qid][doc] = int(grade)
    return dict(qrels)


def build_qrels(queries: list[dict]) -> dict[str, dict[str, int]]:
    """Graded judgments for any scenario-labelled query set, using the same rule as the generator:
    2 = resolved ticket / KB article of the same scenario; 1 = same intent and overlapping products.
    Used to give the dev split (test.jsonl) judgments so retrieval settings can be chosen off the eval set."""
    from scripts.datagen.scenarios import SCENARIOS

    tickets = read_jsonl(PROCESSED_DIR / "tickets.jsonl")
    kb = read_jsonl(PROCESSED_DIR / "kb_articles.jsonl")
    sc_by_id = {s["id"]: s for s in SCENARIOS}
    qrels: dict[str, dict[str, int]] = {}
    for q in queries:
        sc = sc_by_id[q["scenario_id"]]
        grades: dict[str, int] = {}
        for a in kb:
            other = sc_by_id[a["metadata"]["scenario_id"]]
            if other["id"] == sc["id"]:
                grades[a["article_id"]] = 2
            elif other["intent"] == sc["intent"] and set(other["products"]) & set(sc["products"]):
                grades[a["article_id"]] = 1
        for t in tickets:
            m = t["metadata"]
            if not m["resolved"]:
                continue
            if m["scenario_id"] == sc["id"]:
                grades[t["ticket_id"]] = 2
            elif t["category"] == sc["intent"] and set(m["products"]) & set(sc["products"]):
                grades[t["ticket_id"]] = 1
        qrels[q["id"]] = grades
    return qrels


def save_result(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
