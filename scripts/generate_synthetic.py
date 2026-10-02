"""Generate the synthetic telecom dataset from hand-authored scenarios.

Outputs (all JSONL, deterministic for a given --seed):
  data/processed/tickets.jsonl              wave-1 historical tickets (retrieval corpus)
  data/processed/kb_articles.jsonl          wave-1 KB articles
  data/processed/tickets_wave2.jsonl        new ticket class arriving later (evolving data)
  data/processed/kb_articles_wave2.jsonl
  data/evaluation/understanding_eval.jsonl  labelled complaints (intent/product/severity/sentiment)
  data/evaluation/retrieval_eval.jsonl      queries + data/evaluation/retrieval_qrels.tsv (TREC format)
  data/evaluation/generation_eval.jsonl     complaints + reference fix steps
  data/evaluation/groundedness_eval.jsonl   (claim, source, label) triples for Experiment 3
  data/evaluation/novel_intent_eval.jsonl   wave-2 complaints for unknown-intent / evolving-class tests
  data/evaluation/test.jsonl                remaining held-out complaints

Leakage design: evaluation complaints are built ONLY from each scenario's held-out
symptom paraphrases (indices 4-5) and held-out detail sentence, so no eval query
shares its core problem phrasing with any corpus ticket. scripts/verify_splits.py
checks this (exact, near-duplicate and template-family overlap).

Usage: python scripts/generate_synthetic.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.datagen.scenarios import (  # noqa: E402
    AGENT_INTROS,
    AGENT_OUTCOMES,
    AGENT_STEP_PREFIXES,
    CLOSINGS,
    DURATIONS,
    EMOTIONS,
    IMPACTS,
    OPENERS,
    SCENARIOS,
    TIMES,
)

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
CORPUS_SYMPTOMS = slice(0, 4)
HELDOUT_SYMPTOMS = slice(4, 6)

N_UNDERSTANDING = 500
N_RETRIEVAL = 200
N_GENERATION = 100


def bump(severity: str, levels: int = 1) -> str:
    idx = min(SEVERITY_LEVELS.index(severity) + levels, len(SEVERITY_LEVELS) - 1)
    return SEVERITY_LEVELS[idx]


def sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def compose_complaint(rng: random.Random, sc: dict, family: str) -> dict:
    """Build one complaint plus its by-construction labels."""
    if family == "corpus":
        symptom_idx = rng.randrange(0, 4)
        detail_pool = sc["details"][:-1] or sc["details"]
    else:
        symptom_idx = rng.randrange(4, 6)
        detail_pool = sc["details"][-1:]
    symptom = sc["symptoms"][symptom_idx].format(time=rng.choice(TIMES))

    impact_kind = rng.choices(["business", "personal", "none"], weights=[0.2, 0.3, 0.5])[0]
    emotion = rng.choices(["negative", "neutral", "positive"], weights=[0.55, 0.3, 0.15])[0]

    severity = sc["base_severity"]
    if impact_kind == "business":
        severity = bump(severity)

    parts = [rng.choice(OPENERS), sentence(symptom)]
    if rng.random() < 0.6:
        parts.append(sentence(rng.choice(detail_pool)))
    if rng.random() < 0.6:
        parts.append(sentence(f"I've already {rng.choice(sc['tried'])}"))
    if rng.random() < 0.5:
        parts.append(sentence(f"It has been happening {rng.choice(DURATIONS)}"))
    parts.append(rng.choice(IMPACTS[impact_kind]))
    parts.append(rng.choice(EMOTIONS[emotion]))
    parts.append(rng.choice(CLOSINGS))
    complaint = " ".join(p for p in parts if p).strip()

    return {
        "complaint": complaint,
        "category": sc["intent"],
        "products": list(sc["products"]),
        "product": sc["products"][0],
        "severity": severity,
        "sentiment": emotion,
        "template_family": f"{sc['id']}#{'corpus' if family == 'corpus' else 'heldout'}",
        "symptom_variant": symptom_idx,
        "impact": impact_kind,
    }


def compose_resolution(rng: random.Random, sc: dict) -> tuple[str, list[int]]:
    steps = sc["steps"]
    n = rng.randint(min(2, len(steps)), len(steps))
    start = 0 if rng.random() < 0.7 else rng.randint(0, len(steps) - n)
    used = list(range(start, start + n))
    intro = rng.choice(AGENT_INTROS).format(summary=sc["kb_title"].lower())
    prefix = rng.choice(AGENT_STEP_PREFIXES)
    body = " ".join(sentence(steps[i][0]) for i in used)
    cause = f" Root cause: {sc['cause']}" if rng.random() < 0.5 else ""
    return f"{intro}{cause} {prefix}{body} {rng.choice(AGENT_OUTCOMES)}", used


def build_kb_article(sc: dict, idx: int, wave: int) -> dict:
    steps = "\n".join(f"{i}. {sentence(s[0])}" for i, s in enumerate(sc["steps"], 1))
    symptoms = "; ".join(sc["symptoms"][i].format(time="evening") for i in range(2))
    content = (
        f"Applies to: {', '.join(sc['products'])}.\n"
        f"Typical symptoms: {symptoms}.\n"
        f"Cause: {sc['cause']}\n"
        f"Resolution steps:\n{steps}\n"
        f"If these steps do not resolve the issue, escalate to second-line support with the results of each step."
    )
    return {
        "article_id": f"KB-{idx:04d}",
        "title": sc["kb_title"],
        "content": content,
        "category": sc["intent"],
        "product": sc["products"][0],
        "tags": sc["products"],
        "version": 1,
        "source": "synthetic_authored",
        "metadata": {"scenario_id": sc["id"], "wave": wave},
    }


def random_date(rng: random.Random, start: datetime, end: datetime) -> datetime:
    return start + timedelta(seconds=rng.randint(0, int((end - start).total_seconds())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-scenario", type=int, default=25)
    parser.add_argument("--max-per-scenario", type=int, default=90)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    processed = ROOT / "data" / "processed"
    evaluation = ROOT / "data" / "evaluation"
    processed.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)

    tickets: dict[int, list[dict]] = {1: [], 2: []}
    kb: dict[int, list[dict]] = {1: [], 2: []}
    ticket_counter = 0
    wave_windows = {
        1: (datetime(2025, 1, 1), datetime(2026, 6, 30)),
        2: (datetime(2026, 7, 1), datetime(2026, 9, 25)),
    }

    for kb_idx, sc in enumerate(SCENARIOS, 1):
        wave = sc["wave"]
        kb[wave].append(build_kb_article(sc, kb_idx, wave))
        # Skewed volume: some issues are much more common than others.
        n = rng.randint(args.min_per_scenario, args.max_per_scenario)
        for _ in range(n):
            ticket_counter += 1
            c = compose_complaint(rng, sc, "corpus")
            unresolved = rng.random() < 0.04
            if unresolved:
                used: list[int] = []
                resolution = "Customer did not respond to follow-up. Closed without confirmed resolution."
            else:
                resolution, used = compose_resolution(rng, sc)
            created = random_date(rng, *wave_windows[wave])
            tickets[wave].append(
                {
                    "ticket_id": f"TKT-{ticket_counter:06d}",
                    "complaint": c["complaint"],
                    "resolution": resolution,
                    "category": c["category"],
                    "product": c["product"],
                    "severity": c["severity"],
                    "sentiment": c["sentiment"],
                    "created_at": created.isoformat(),
                    "resolved_at": (created + timedelta(hours=rng.randint(1, 96))).isoformat(),
                    "source": "synthetic_template",
                    "label_source": "generator_by_construction",
                    "split": "retrieval_corpus",
                    "metadata": {
                        "scenario_id": sc["id"],
                        "products": c["products"],
                        "template_family": c["template_family"],
                        "steps_used": used,
                        "resolved": not unresolved,
                        "wave": wave,
                    },
                }
            )

    # ---------------- held-out evaluation complaints (wave 1) --------------
    wave1 = [s for s in SCENARIOS if s["wave"] == 1]
    total_eval = N_UNDERSTANDING + N_RETRIEVAL + N_GENERATION + 200
    eval_items = []
    for i in range(total_eval):
        sc = wave1[i % len(wave1)]  # balanced across scenarios
        c = compose_complaint(rng, sc, "heldout")
        c["scenario_id"] = sc["id"]
        eval_items.append(c)
    rng.shuffle(eval_items)
    seen, unique_items = set(), []
    for item in eval_items:
        if item["complaint"] not in seen:
            seen.add(item["complaint"])
            unique_items.append(item)
    eval_items = unique_items

    def take(n, prefix):
        nonlocal eval_items
        chunk, eval_items = eval_items[:n], eval_items[n:]
        for j, item in enumerate(chunk, 1):
            item["id"] = f"{prefix}-{j:04d}"
        return chunk

    understanding = take(N_UNDERSTANDING, "UND")
    retrieval = take(N_RETRIEVAL, "RET")
    generation = take(N_GENERATION, "GEN")
    test = take(len(eval_items), "TST")

    # ---------------- retrieval qrels (TREC: qid 0 docid grade) ------------
    kb_by_scenario = {a["metadata"]["scenario_id"]: a for a in kb[1]}
    sc_by_id = {s["id"]: s for s in SCENARIOS}
    qrels_lines = []
    for q in retrieval:
        sc = sc_by_id[q["scenario_id"]]
        qrels_lines.append(f"{q['id']} 0 {kb_by_scenario[sc['id']]['article_id']} 2")
        for t in tickets[1]:
            m = t["metadata"]
            if m["scenario_id"] == sc["id"] and m["resolved"]:
                qrels_lines.append(f"{q['id']} 0 {t['ticket_id']} 2")
            elif t["category"] == sc["intent"] and set(m["products"]) & set(sc["products"]) and m["resolved"]:
                qrels_lines.append(f"{q['id']} 0 {t['ticket_id']} 1")
        for a in kb[1]:
            other = sc_by_id[a["metadata"]["scenario_id"]]
            if (
                other["id"] != sc["id"]
                and other["intent"] == sc["intent"]
                and set(other["products"]) & set(sc["products"])
            ):
                qrels_lines.append(f"{q['id']} 0 {a['article_id']} 1")

    for g in generation:
        sc = sc_by_id[g["scenario_id"]]
        g["reference_steps"] = [s[0] for s in sc["steps"]]
        g["reference_kb_id"] = kb_by_scenario[sc["id"]]["article_id"]

    # ---------------- groundedness triples for Experiment 3 ----------------
    grounded = []
    for sc in wave1:
        kb_art = kb_by_scenario[sc["id"]]
        resolved = [t for t in tickets[1] if t["metadata"]["scenario_id"] == sc["id"] and t["metadata"]["steps_used"]]
        ticket = rng.choice(resolved)
        step_in_ticket = rng.choice(ticket["metadata"]["steps_used"])
        kb_step = rng.randrange(len(sc["steps"]))
        others = [s for s in wave1 if s["id"] != sc["id"] and s["intent"] == sc["intent"]] or [
            s for s in wave1 if s["id"] != sc["id"]
        ]
        other = rng.choice(others)
        ticket_text = f"{ticket['complaint']}\n\nResolution: {ticket['resolution']}"
        kb_text = f"{kb_art['title']}\n\n{kb_art['content']}"
        grounded += [
            {
                "claim": sc["steps"][kb_step][1],
                "source": kb_text,
                "source_type": "kb_article",
                "label": "supported",
                "scenario_id": sc["id"],
            },
            {
                "claim": sc["steps"][step_in_ticket][1],
                "source": ticket_text,
                "source_type": "ticket",
                "label": "supported",
                "scenario_id": sc["id"],
            },
            {
                "claim": rng.choice(other["steps"])[0],
                "source": kb_text,
                "source_type": "kb_article",
                "label": "unsupported",
                "scenario_id": sc["id"],
            },
            {
                "claim": sc["contradiction"],
                "source": rng.choice([kb_text, ticket_text]),
                "source_type": "mixed",
                "label": "contradicted",
                "scenario_id": sc["id"],
            },
        ]
    for j, g in enumerate(grounded, 1):
        g["id"] = f"GRD-{j:04d}"

    # ---------------- novel-intent (wave 2) evaluation ---------------------
    novel = []
    for i in range(60):
        sc = [s for s in SCENARIOS if s["wave"] == 2][i % 3]
        c = compose_complaint(rng, sc, "heldout")
        c["scenario_id"] = sc["id"]
        c["id"] = f"NOV-{i + 1:04d}"
        novel.append(c)

    def dump(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(processed / "tickets.jsonl", tickets[1])
    dump(processed / "kb_articles.jsonl", kb[1])
    dump(processed / "tickets_wave2.jsonl", tickets[2])
    dump(processed / "kb_articles_wave2.jsonl", kb[2])
    dump(evaluation / "understanding_eval.jsonl", understanding)
    dump(evaluation / "retrieval_eval.jsonl", retrieval)
    dump(evaluation / "generation_eval.jsonl", generation)
    dump(evaluation / "groundedness_eval.jsonl", grounded)
    dump(evaluation / "novel_intent_eval.jsonl", novel)
    dump(evaluation / "test.jsonl", test)
    (evaluation / "retrieval_qrels.tsv").write_text("\n".join(qrels_lines) + "\n", encoding="utf-8")

    print(f"seed={args.seed}")
    print(f"wave-1 tickets: {len(tickets[1])}  KB: {len(kb[1])}")
    print(f"wave-2 tickets: {len(tickets[2])}  KB: {len(kb[2])}")
    print(
        f"eval: understanding={len(understanding)} retrieval={len(retrieval)} generation={len(generation)} "
        f"test={len(test)} groundedness={len(grounded)} novel={len(novel)}"
    )
    print("label dist (understanding):", dict(Counter(u["category"] for u in understanding)))
    print("qrels lines:", len(qrels_lines))


if __name__ == "__main__":
    main()
